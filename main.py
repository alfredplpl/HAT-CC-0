import os.path

import torch
from PIL import Image
from torchvision import transforms
from hat import HAT
import argparse
import os
import requests

parser = argparse.ArgumentParser(description='Super resolution.')
parser.add_argument('device', type=str,
                    help='A device for super resolution: cpu or cuda')
parser.add_argument('image_path', type=str,
                    help='A path to your image.')
parser.add_argument('output_path', type=str,
                    help='A path to your image.')
args = parser.parse_args()


load_path="hat_cc_0.pth"
if(not os.path.exists(load_path)):
    print("Not found the hat model. The download start.")
    res=requests.get("https://storage.googleapis.com/distributed-models/hat_cc_0.pth")
    with open(load_path,"wb") as f:
        f.write(res.content)

net = HAT(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv"
)

low_img=Image.open(args.image_path)
if(args.image_path=="sample.jpg"):
    low_img=low_img.crop((172, 322, 428, 578)) # center crop 256x256
low_tensor=transforms.ToTensor()(low_img)

weight = torch.load(load_path, map_location="cpu")
net.load_state_dict(weight["params_ema"])
net.eval()

net.to(args.device)
low_tensor=low_tensor.to(args.device)

with torch.no_grad():
    high_tensor = net(low_tensor)
    high_tensor=torch.clamp(high_tensor,0.0,1.0)

high_img=transforms.ToPILImage()(high_tensor[0])
high_img.save(args.output_path)
print(f"Output the image as {args.output_path}")
