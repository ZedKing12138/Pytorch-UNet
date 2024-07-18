# %%
#modify unet（adapted for dynamic）
#input： 1*3*144*176*144
#        t_emd
#output： 1*3*144*176*144



# %%
#test pipeline datasize changes
#input：1*1*64


# %%
import torch
import os
from unet.unet_model import UNet_3D_ebd

device = "cuda:0"
input_x = torch.rand(size=(1,3,144,176,144)).to(device)

model = UNet_3D_ebd(n_channels=4,n_classes=3,trilinear=True).to(device)
out_mask = model(1.0,input_x)


print(out_mask.shape)


