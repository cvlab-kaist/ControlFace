# *************************************************************************
# Copyright (2024) Bytedance Inc.
#
# Copyright (2024) LightningDrag Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from src.models.transformer_2d import Transformer2DModel
class ConvBlock(nn.Module):

    def __init__(self,
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding,
                bias=False,
                ):
        super().__init__()
        # possibly downsample at the first conv
        self.conv1 = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.ac = nn.SiLU()
        # maintain the original shape at the second conv
        self.conv2 = nn.Conv2d(out_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac(x)
        x = self.conv2(x)
        x = self.ac(x)
        return x

class AbstractPointEmbedding(ModelMixin):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

# increase number of channels in every downsample block

class PointEmbeddingModel_CrossAttn(AbstractPointEmbedding):
    def __init__(
        self,
        input_dim=2,
        vae_downsample_scale=8,
        unet_downsample_scale=8,
        embed_dim=16,
        bias=False,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(input_dim, embed_dim,
                       kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_in_cond = nn.Conv2d(input_dim, embed_dim,
                       kernel_size=3, stride=1, padding=1, bias=bias)
        in_dim = embed_dim
        self.downsample_blocks_vae = nn.Sequential(*[
            ConvBlock((2**i)*in_dim, (2**(i+1))*in_dim,
            kernel_size=3, stride=2, padding=1, bias=bias)
            for i in range(int(np.log2(vae_downsample_scale)))
        ])

        self.downsample_blocks_vae_cond = nn.Sequential(*[
            ConvBlock((2**i)*in_dim, (2**(i+1))*in_dim,
            kernel_size=3, stride=2, padding=1, bias=bias)
            for i in range(int(np.log2(vae_downsample_scale)))
        ])
        self.mid = Transformer2DModel(8,in_dim*vae_downsample_scale//8,in_dim*vae_downsample_scale,cross_attention_dim=in_dim*vae_downsample_scale)

        self.downsample_blocks_unet = nn.ModuleList([
            ConvBlock((2**i)*in_dim*vae_downsample_scale,
                      (2**(i+1))*in_dim*vae_downsample_scale,
                      kernel_size=3, stride=2, padding=1, bias=bias)
            for i in range(int(np.log2(unet_downsample_scale)))
        ])

        self.downsample_blocks_unet_cond = nn.ModuleList([
            ConvBlock((2**i)*in_dim*vae_downsample_scale,
                      (2**(i+1))*in_dim*vae_downsample_scale,
                      kernel_size=3, stride=2, padding=1, bias=bias)
            for i in range(int(np.log2(unet_downsample_scale)))
        ])

        self.downsample_blocks_unet_trans = nn.ModuleList([
            Transformer2DModel(8,(2**(i+1))*in_dim//8,(2**(i+1))*in_dim*vae_downsample_scale,cross_attention_dim=(2**(i+1))*in_dim*vae_downsample_scale)
            for i in range(int(np.log2(unet_downsample_scale)))
        ])
        self.embed_dim = embed_dim
        # everytime downsample 2x, double the number of channels
        self.output_dim = [embed_dim*vae_downsample_scale*(2**i) \
            for i in range(1 + int(np.log2(unet_downsample_scale)))]

    def forward(self, src, cond):
        # concat on channel
        
        embedding = self.conv_in(src)
        embedding = self.downsample_blocks_vae(embedding)

        embedding_cond = self.conv_in_cond(cond)
        embedding_cond = self.downsample_blocks_vae_cond(embedding_cond)
        b,c,h,w = embedding_cond.shape
        embedding_cond_input = embedding_cond.permute(0, 2, 3, 1).reshape(
                b, h*w, c
            )
        embedding, _ = self.mid(hidden_states=embedding,
        encoder_hidden_states=embedding_cond_input,return_dict=False)

        all_embeddings = [embedding]
        for module,module_cond,module_trans in zip(self.downsample_blocks_unet,self.downsample_blocks_unet_cond,self.downsample_blocks_unet_trans):
            embedding = module(embedding)
            embedding_cond = module_cond(embedding_cond)
            b,c,h,w = embedding_cond.shape
            embedding_cond_input = embedding_cond.permute(0, 2, 3, 1).reshape(
                b, h*w, c
            )
            embedding,_ = module_trans(hidden_states=embedding,
        encoder_hidden_states=embedding_cond_input,return_dict=False)
            all_embeddings.append(embedding)

        return all_embeddings