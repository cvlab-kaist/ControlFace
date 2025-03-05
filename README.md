<p align="center">
  <h1 align="center">ControlFace: Harnessing Facial Parametric Control for Face Rigging</h1>
  <h3 align="center">CVPR 2025</h3>
  <p align="center">
     <a>Wooseok Jang</a>
    ·
     <a>Youngjun Hong</a>
    ·
    <a>Geonho Cha</a>
    ·
    <a>Seungryong Kim</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.01160">Paper </a> | <a href="https://cvlab-kaist.github.io/ControlFace">Project Page </a> </h3>
  <div align="center"></div>
  <img src="videos/caption.png" width="100%">
  <img src="videos/1.gif" width="100%">
  <img src="videos/2.gif" width="100%">
  <img src="videos/3.gif" width="100%">
  <img src="videos/4.gif" width="100%">
</p>

## 1. Environment setup
Build the environment as follows:
```bash
conda create -n controlface python=3.8
conda activate controlface

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

conda install mpi4py dlib scikit-learn scikit-image tqdm -c conda-forge

pip install -r requirements.txt
```

## 2. Download pretrained weights
First run the following command which will automatically download the weights.
Weights will be placed under the `./pretrained_weights` directory. 
```bash
python tools/download_weights.py
```
Then follow the `DECA Setup` stage present in [here](https://github.com/adobe-research/diffusion-rig).

## 3. Inference
We provide a example script for face editing. Change the command below to specify the attribute you want to edit (pose, expression, light, shape) by modifying the --mode flag.
```bash
PATH_TO_REFERENCE="./examples/00013.png"
PATH_TO_TARGET="./examples/00690.png"
python sample.py --ref ${PATH_TO_REFERENCE} \
 --tgt ${PATH_TO_TARGET} \
 --mode pose
```
The output will be saved under the `./output` directory.

## Acknowledgements
Our project builds upon and incorporates elements from [DiffusionRig](https://github.com/adobe-research/diffusion-rig), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), and [LightningDrag](https://github.com/magic-research/LightningDrag). We would like to thank the authors and maintainers of these projects for their invaluable work and for making their code available to the community.
