import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
from io import BytesIO

import lmdb
import pickle
import numpy as np
import time
from transformers import CLIPImageProcessor
from src.utils.util import get_fps, read_frames
from decord import VideoReader, gpu
from torchvision.utils import save_image
from typing import List


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        path_arc=None,
    ):
        super().__init__()
        self.zfill = 6

        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if path_arc is not None:
            self.arc_env = lmdb.open(
                path_arc,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            self.arc_env = None

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):


        tgt_img_idx = random.randint(0, self.length-1)

        with self.env.begin(write=False) as txn:
            key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
            image_bytes = txn.get(key)
            
            key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_albedo_bytes = txn.get(key)
            
            key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_rendered_bytes = txn.get(key)


            key = f"image_{str(tgt_img_idx).zfill(self.zfill)}".encode("utf-8")
            tgt_image_bytes = txn.get(key)

            key = f"normal_{str(tgt_img_idx).zfill(self.zfill)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(tgt_img_idx).zfill(self.zfill)}".encode("utf-8")
            albedo_bytes = txn.get(key)
            
            key = f"rendered_{str(tgt_img_idx).zfill(self.zfill)}".encode("utf-8")
            rendered_bytes = txn.get(key)
        

        buffer = BytesIO(normal_bytes)
        normal = pickle.load(buffer)

        buffer = BytesIO(albedo_bytes)
        albedo = pickle.load(buffer)

        buffer = BytesIO(rendered_bytes)
        rendered = pickle.load(buffer)

        control = torch.cat([rendered, normal, albedo], dim=0)

        buffer = BytesIO(src_normal_bytes)
        src_normal = pickle.load(buffer)

        buffer = BytesIO(src_albedo_bytes)
        src_albedo = pickle.load(buffer)

        buffer = BytesIO(src_rendered_bytes)
        src_rendered = pickle.load(buffer)

        control_src = torch.cat([src_rendered ,src_normal, src_albedo], dim=0)
        mask = ~((control_src <= 0 ).all(dim=0))


        tgt_mask = ~((control <= 0 ).all(dim=0))

        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)
        buffer = BytesIO(tgt_image_bytes)
        tgt_image = Image.open(buffer)
        ref_image = self.transform(image)
        tgt_image = self.transform(tgt_image)
        clip_image = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values[0]

        return {
        "tgt_img": tgt_image,
        "ref_img": ref_image,
        "clip_img": clip_image,
        "control": control,
        "control_src":control_src,
        "mask": mask,
        "tgt_mask": tgt_mask,
    }

class ImageDataset_FFHQ(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        path_arc=None,
    ):
        super().__init__()
        self.zfill = 6

        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if path_arc is not None:
            self.arc_env = lmdb.open(
                path_arc,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            self.arc_env = None

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()
        

    def __len__(self):
        return self.length

    def __getitem__(self, index):


        with self.env.begin(write=False) as txn:
            key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
            image_bytes = txn.get(key)
            
            key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_albedo_bytes = txn.get(key)
            
            key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
            src_rendered_bytes = txn.get(key)

        buffer = BytesIO(src_normal_bytes)
        src_normal = pickle.load(buffer)

        buffer = BytesIO(src_albedo_bytes)
        src_albedo = pickle.load(buffer)

        buffer = BytesIO(src_rendered_bytes)
        src_rendered = pickle.load(buffer)

        control_src = torch.cat([src_rendered ,src_normal, src_albedo], dim=0)





        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)
        
        ref_image = self.transform(image)

        clip_image = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values[0]

        return {
        "tgt_img": ref_image,
        "ref_img": ref_image,
        "clip_img": clip_image,
        "control": control_src,
        "control_src":control_src,
    }

class ImageDataset_Video_dep(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
    ):
        super().__init__()
        self.zfill = 6

        self.path = path

        self.env = lmdb.open(f"./VoxCelebCode.lmdb",max_readers=32,readonly=True,lock=False,
                            #  readahead=False,
                             meminit=False
                             )

        # [
        # lmdb.open(
        #     f"./celeb_{str(i)}.lmdb",
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # ) for i in range(8)]
        with open("/mnt/video-nfs5/datasets/celebvhq/train_datalist.json",'r') as train:
            self.dl=json.load(train)
    
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        # with self.env.begin(write=False) as txn:
        self.length = len(self.dl)

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()

        self.video_root = "./pose_vid/original"
        self.sample_margin = 30
        self.keys=['shape', 'tex', 'exp', 'pose', 'cam', 'light','tform','images']
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_id=self.dl[index]
        video = read_frames(os.path.join(self.video_root,video_id+'.mp4'))
        video_length = len(video)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        i = index%8
        control={}
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = f"{k}_{str(index).zfill(6)}".encode("utf-8")
                code_bytes = txn.get(key)
                buffer = BytesIO(code_bytes)
                control[k] = pickle.load(buffer)[tgt_img_idx]
        

        src_image = video[ref_img_idx]
        tgt_image = video[tgt_img_idx]
        ref_image = self.transform(src_image)
        tgt_image = self.transform(tgt_image)
        clip_image = self.processor(
            images=src_image, return_tensors="pt"
        ).pixel_values[0]

        return {
        "ref_img": ref_image,
        "clip_img": clip_image,
        "control": control,
        "tgt_img": tgt_image
    }

class ImageDataset_Video(torch.utils.data.Dataset):
    def __init__(
        self,
        path,num,
        arc=None
    ):
        super().__init__()
        self.zfill = 6

        self.path = path

        # with open("/mnt/video-nfs5/datasets/celebvhq/train_datalist.json",'r') as train:
        self.dl=os.listdir("pose_vid_new/original")
        # with self.env.begin(write=False) as txn:
        self.length = len(self.dl)

        self.arc=arc

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()
        self.num=num
        self.video_root = "pose_vid_new/original"
        self.sample_margin = 30
        self.keys=['shape', 'tex', 'exp', 'pose', 'cam', 'light','tform','images']
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_id=self.dl[index][:-4]

        video = VideoReader(os.path.join(self.video_root,video_id+'.mp4'))

        video_length = len(video)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)
        
        src_image = Image.fromarray(video[ref_img_idx].asnumpy())
        tgt_image_ = Image.fromarray(video[tgt_img_idx].asnumpy())
        del video
        ref_image = self.transform(src_image)
        tgt_image = self.transform(tgt_image_)
        clip_image = self.processor(
            images=src_image, return_tensors="pt"
        ).pixel_values[0]
        tgt_clip_img = self.processor(
            images=tgt_image_, return_tensors="pt"
        ).pixel_values[0]

        control = torch.load(os.path.join(self.path,f"{video_id}.pt"),map_location='cpu')
        control.pop('images',None)
        control_src={}
        for k,v in control.items():
            control[k] = v[tgt_img_idx]
            control_src[k] = v[ref_img_idx]

        return {
        "ref_img": ref_image,
        "clip_img": clip_image,
        "tgt_clip_img": tgt_clip_img,
        "control": control,
        "control_src": control_src,
        "tgt_img": tgt_image
    }

class VideoDataset_Video(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        arc=None
    ):
        super().__init__()
        self.zfill = 6

        self.path = path
        self.dl = os.listdir('pose_vid_new/original')
        # with self.env.begin(write=False) as txn:
        self.length = len(self.dl)

        self.arc=arc

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.sample_rate = 4
        self.n_sample_frames = 24
        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()

        self.video_root = "pose_vid_new/original"
        self.sample_margin = 30
        self.keys=['shape', 'tex', 'exp', 'pose', 'cam', 'light','tform','images']
    def __len__(self):
        return self.length
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):


        video_id=self.dl[index][:-4]

        video = VideoReader(os.path.join(self.video_root,video_id+'.mp4'))

        video_length = len(video)

        
        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        )
        control = torch.load(os.path.join(self.path,f"{video_id}.pt"),map_location='cpu')
        control.pop('images',None)
        control_src={}
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video[ref_img_idx].asnumpy())

        for k,v in control.items():
            control_src[k] = v[ref_img_idx]

        control = {key: value[batch_index] for key, value in control.items()}

        vid_pil_image_list = []

        for index in batch_index:
            img = video[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))

        del video
        clip_image = self.processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]
        ref_img = self.augmentation(ref_img,self.transform)
        pix_values_vid = self.augmentation(vid_pil_image_list,self.transform)

        

        return {
        "ref_img": ref_img,
        "clip_img": clip_image,
        "control": control,
        "control_src": control_src,
        "pixel_values_vid": pix_values_vid,
    }


class ImageDataset_Video_Top5(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        arc=None
    ):
        super().__init__()
        self.zfill = 6

        self.path = path

        with open("/mnt/video-nfs5/datasets/celebvhq/train_datalist.json",'r') as train:
            self.dl=json.load(train)
        self.length = len(self.dl)

        self.arc=arc

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.processor = CLIPImageProcessor()

        self.video_root = "./pose_vid/original"
        self.sample_margin = 30
        self.keys=['shape', 'exp', 'pose', 'light']
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_id = self.dl[index]
        video = VideoReader(os.path.join(self.video_root, video_id + '.mp4'))
        video_length = len(video)

        # Randomly pick the source frame index
        ref_img_idx = random.randint(0, video_length - 1)

        # Source image
        src_image = Image.fromarray(video[ref_img_idx].asnumpy())

        # Load control file
        control = torch.load(os.path.join(self.path, f"code_{str(index).zfill(6)}.pt"), map_location='cpu')
        control.pop('images', None)  # Remove unnecessary key if it exists

        # Randomly choose one of the control keys (e.g., 'light', 'exp', 'pose')
        selected_key = random.choice(self.keys)
        
        # Retrieve control values for the selected key (tensor)
        control_values = torch.tensor(control[selected_key], dtype=torch.float32)  # Shape: (video_length, control_dim)
        if selected_key == "light":
            control_values = control_values.reshape(-1,27)
        # Source control for the selected frame, keeping all keys
        control_src = {k: v[ref_img_idx] for k, v in control.items()}  # All keys for the source control

        # Efficiently compute distances based on the selected key
        distances = torch.norm(control_values - control_values[ref_img_idx], dim=-1)  # Compare against the selected key
        # distances[ref_img_idx] = -float('inf')  # Exclude source frame from distance computation

        # Get the top 5 indices with the furthest distances
        top_k = 10
        top_k_indices = torch.topk(distances, top_k).indices.tolist()

        # Use a set operation for faster computation of remaining indices
        top_k_set = set(top_k_indices)
        remaining_indices = [i for i in range(video_length) if i not in top_k_set]
        # Randomly select one from the remaining indices
        tgt_img_idx_not = random.choice(remaining_indices)

        # Randomly select one from the top_k indices
        tgt_img_idx = random.choice(top_k_indices)

        # # Randomly select one from the top 5
        # tgt_img_idx = random.choice(top_k_indices)

        # Get both reference and target images before deleting the video
        tgt_image_ = Image.fromarray(video[tgt_img_idx].asnumpy())

        tgt_image_not = Image.fromarray(video[tgt_img_idx_not].asnumpy())

        # Delete video to save memory after extracting frames
        del video

        # Transform the images
        ref_image = self.transform(src_image)
        tgt_image = self.transform(tgt_image_)
        tgt_image_not = self.transform(tgt_image_not)

        # Control for the target frame (keeping all keys)
        control_tgt = {k: v[tgt_img_idx] for k, v in control.items()}  # Load control values for all keys
        control_tgt_not = {k: v[tgt_img_idx_not] for k, v in control.items()}  # Load control values for all keys
        return {
            "ref_img": ref_image,
            "clip_img": self.processor(images=src_image, return_tensors="pt").pixel_values[0],
            "control": control_tgt,  # All control values for target frame
            "control_src": control_src,  # Source control with all keys
            "tgt_img": tgt_image,
            "tgt_img_not": tgt_image_not,
            "control_not": control_tgt_not,
        }



       
class ImageDataset_Arc(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        arc_path,
        path_code=None,
        stage=1,
    ):
        super().__init__()
        self.stage=stage
        self.zfill = 6
        
        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.arc_env = lmdb.open(
            arc_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        self.idxs = [*range(self.length)]

        self.modes=["light","pose","exp"]
        
        

        if stage == 2:
        
            if path_code is None:
                path_code = path
            self.code_env = lmdb.open(
            path_code,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.stage == 1:

            with self.env.begin(write=False) as txn:
                key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
                image_bytes = txn.get(key)

                key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
                normal_bytes = txn.get(key)

                key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
                albedo_bytes = txn.get(key)
                
                key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
                rendered_bytes = txn.get(key)

            with self.arc_env.begin(write=False) as txn:  
                key = f"arc_image_{str(index).zfill(self.zfill)}".encode("utf-8")
                arc_image_bytes = txn.get(key) 

            buffer = BytesIO(normal_bytes)
            normal = pickle.load(buffer)

            buffer = BytesIO(albedo_bytes)
            albedo = pickle.load(buffer)

            buffer = BytesIO(rendered_bytes)
            rendered = pickle.load(buffer)
            
            control = torch.cat([rendered, normal, albedo], dim=0)
            
            buffer = BytesIO(image_bytes)
            image = Image.open(buffer)

            buffer = BytesIO(arc_image_bytes)
            arc_image = Image.open(buffer)
            arc_image = self.transform(arc_image)
            # faces = np.array(image)[:,:,::-1]
            # with torch.no_grad():
                # faces = self.app.get(faces)
                # faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                # id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)
            
            
            image = self.transform(image)
            
            
            return {
            "image": image,
            "arc_image": arc_image,
            "control": control,
        }
        
        else:
        
            # index_=random.choice([n for n in self.idxs if n!= index])
            # mode=random.choice(self.modes)
            with self.env.begin(write=False) as txn:
                key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
                image_bytes = txn.get(key)

                key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
                normal_bytes = txn.get(key)

                key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
                albedo_bytes = txn.get(key)

                key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
                rendered_bytes = txn.get(key)
            with self.code_env.begin(write=False) as txn:
                key = f"code_{str(index).zfill(self.zfill)}".encode("utf-8")
                code_bytes = txn.get(key)
            
            buffer = BytesIO(image_bytes)
            image0 = Image.open(buffer)
            

            buffer = BytesIO(normal_bytes)
            normal = pickle.load(buffer)

            buffer = BytesIO(albedo_bytes)
            albedo = pickle.load(buffer)

            buffer = BytesIO(rendered_bytes)
            rendered = pickle.load(buffer)
            control_0 = torch.cat([rendered, normal, albedo], dim=0)

            buffer = BytesIO(code_bytes)
            code_0 = pickle.load(buffer)
            
            # for key in code_0.keys():
            #     code_0[key]=code_0[key].unsqueeze(dim=0).to('cuda')
            # breakpoint()
            image_0 = self.transform(image0)
            image0=torch.tensor(np.array(image0)/255).permute(2,0,1)

           
            
            return{
                "image_0": image_0,
                "control_0": control_0,
                "code_0": code_0,
                "img0": image0,
            }
#