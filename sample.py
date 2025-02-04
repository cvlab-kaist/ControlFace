import argparse

import os

import torch

from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection

from src.models.mutual_self_attention_point import set_up_point_attn_processor
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.pipelines.pipeline_point import Pose2ImagePipeline_Point_CFG
from src.models.pointemb import PointEmbeddingModel_CrossAttn

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset

from torchvision.utils import save_image

def create_inter_data(deca,dataset,mode):


    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    tform = dataset[-1]["tform"].unsqueeze(0)
    tform = torch.inverse(tform).transpose(1, 2).to("cuda")
    with torch.no_grad():
        code2 = deca.encode(img2)
        code2['tform']=tform
        
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")


    img1 = dataset[0]["image"].unsqueeze(0).to("cuda")

    with torch.no_grad():
        code1 = deca.encode(img1)
   
    tform = dataset[0]["tform"].unsqueeze(0)
    tform = torch.inverse(tform).transpose(1, 2).to("cuda")
    original_image = dataset[0]["original_image"].unsqueeze(0).to("cuda")
    code1["tform"] = tform

    batch={}
    control=[]
    ref_image=[]
    target_image=[]
    ref_control = []
    with torch.no_grad():
        opdict_, _ = deca.decode(
            code1,
            original_image=original_image,
            render_orig=True,
            tform=code1["tform"],
        )


    code = {}
    for k in code1:
        code[k] = code1[k].clone()

    if mode == 'pose':
        code["pose"][:, :3] = code2["pose"][:, :3]
    elif mode == 'exp':
        code["exp"] = code2["exp"]
        code["pose"][:, 3:] = code2["pose"][:, 3:]
    elif mode == 'light':
        code['light'] = code2['light']
    elif mode == 'light':
        code['light'] = code2['light']
    elif mode == 'shape':
        code["shape"] = code2['shape']
    else:
        print(f'please check the mode: {mode}')
        exit()
        
    opdict, _ = deca.decode(
        code,
        render_orig=True,
        original_image=original_image,
        tform=code["tform"]
    )
    

    
    rendered = opdict["rendered_images"].detach().squeeze()
    normal = opdict["normal_images"].detach().squeeze()
    albedo = opdict["albedo_images"].detach().squeeze()
    rendered_ = opdict_["rendered_images"].detach().squeeze()
    normal_ = opdict_["normal_images"].detach().squeeze()
    albedo_ = opdict_["albedo_images"].detach().squeeze()
    
    ref_control.append(torch.cat([rendered_,normal_,albedo_],dim=0)) 
    control.append(torch.cat([rendered,normal,albedo],dim=0))
    ref_image.append(original_image.squeeze())
    target_image.append(image2.squeeze())

    batch['control']=torch.stack(control,dim=0)
    batch['ref_image']=torch.stack(ref_image,dim=0)
    batch['target_image']=torch.stack(target_image,dim=0)
    batch['ref_control']=torch.stack(ref_control,dim=0)
    return batch

def load_model(args):
    
    cfg = OmegaConf.load(args.conf)
    
    # Load DECA and model components
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca = DECA(config=deca_cfg, device="cuda")
    
    # Load models
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to("cuda")
    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet").to("cuda")
    denoising_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet").to("cuda")
    pose_guider = PoseGuider(conditioning_embedding_channels=320, conditioning_channels=9, block_out_channels=(16, 32, 96, 256)).to("cuda")
    point_emb = PointEmbeddingModel_CrossAttn(input_dim=9, embed_dim=16).to("cuda")

    # Set up attention processor in the denoising unet

    set_up_point_attn_processor(denoising_unet, point_emb)
    
    # Load weights
    
    point_emb.load_state_dict(torch.load(os.path.join(cfg.controlface_path, f'point_embedding.pth'),map_location="cpu"))
    pose_guider.load_state_dict(torch.load(os.path.join(cfg.controlface_path, f'pose_guider.pth'),map_location="cpu"))
    denoising_unet.load_state_dict(torch.load(os.path.join(cfg.controlface_path, f'denoising_unet.pth'),map_location="cpu"))
    reference_unet.load_state_dict(torch.load(os.path.join(cfg.controlface_path, f'reference_unet.pth'),map_location="cpu"), strict=False)
    # Set up scheduler and pipeline
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )

    scheduler = DDIMScheduler(**sched_kwargs)
    
    pipe = Pose2ImagePipeline_Point_CFG(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        point_embed=point_emb,
        scheduler=scheduler,
    )
    
    return deca, pipe

def infer(pipe, deca, args,):
    dataset = deca_dataset.TestData([args.ref, args.tgt], iscrop=True, size=256)
    x = create_inter_data(deca, dataset,args.mode)
    
    ref_image = x['ref_image']
    physic_cond_ref = x['ref_control']
    physic_cond_tgt = x['control']
    hdl_pts = physic_cond_ref
    tgt_pts = physic_cond_tgt
   
    images = pipe(
        ref_image,
        physic_cond_tgt,
        256,
        256,
        20,
        args.scale,
        hdl_pts=hdl_pts,
        tgt_pts=tgt_pts,
        source_pose_image=physic_cond_ref,
    ).images
    os.makedirs(args.output_path,exist_ok=True)
    name = (args.ref).split('/')[-1]
    name = name.split('.')[0]
    output_name= os.path.join(args.output_path,f'result_{name}_{args.mode}.png')  
    save_image(images,output_name)
        
    print(f"Inference completed and saved to {output_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf", type=str, default='./configs/inference.yaml',
    )
    parser.add_argument(
        "--ref", type=str, 
    )
    parser.add_argument(
        "--tgt", type=str,
    )
    parser.add_argument(
        "--mode", type=str,
    )
    parser.add_argument(
        "--output_path", type=str, default='./output'
    )
    parser.add_argument(
        "--scale", type=float, default=3.0
    )
    args = parser.parse_args()
    
    deca, pipe = load_model(args)


    infer(pipe, deca, args)

if __name__ == "__main__":
    main()