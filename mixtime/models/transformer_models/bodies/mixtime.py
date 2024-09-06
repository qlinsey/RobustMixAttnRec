from torch import nn as nn
import torch
from .transformers.transformer_mixtime import TransformerMixtimeBlock

# from .transformers.GaussianBlock import GaussianBlock
from .transformers.GaussianNoise import GaussianNoise


class MixtimeBody(nn.Module):
    def __init__(self, args, La, Lr):
        super().__init__()
        print("in MixtimeBody args.noise_reg=", args.noise_reg)
        n_layers = args.num_blocks
        print("n_layers=", n_layers, "la=", La, "lr=", Lr)
        sigma = args.sigma
        self.gaussian_noises = GaussianNoise(sigma)
        self.transformer_blocks = nn.ModuleList(
            [TransformerMixtimeBlock(args, La, Lr) for _ in range(n_layers)]
        )

    def forward(self, noise_reg, x, attn_mask, abs_kernel, rel_kernel, info):
        # x : B x T x H
        # abs_kernel : La of [B x T x H]
        # rel_kernel : Lr of [B x T x T x H]
        # print("x.shape=",x.shape)
        print("here noise_reg=", noise_reg)
        if self.training and noise_reg:
            x = self.gaussian_noises(x)
            """
            abs_kernels=[]
            for i in range(len(abs_kernel)):
                #print("before gau abs_kernel[i].shape=",abs_kernel[i].shape) 
                #x = torch.cat((x, abs_kernel[i]), dim=-1)
                abs_kernels.append(self.gaussian_noises(abs_kernel[i]))
                #print("after gau abs_kernel[i].shape=",abs_kernels[i].shape)
            rel_kernels=[]
            for i in range(len(rel_kernel)):
                #print("before gau rel_kernel[i].shape=",rel_kernel[i].shape) 
                rel_kernels.append(self.gaussian_noises(rel_kernel[i]))
                #x = torch.cat((x, rel_kernel[i]), dim=-1)
            """
        """
        else:
            abs_kernels = abs_kernel
        """
        for layer, transformer in enumerate(self.transformer_blocks):
            x = transformer.forward(
                x, attn_mask, abs_kernel, rel_kernel, layer=layer, info=info
            )
        return x
