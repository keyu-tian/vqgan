"""
git clone https://github.com/keyu-tian/vqgan.git
cd vqgan/scripts
mkdir -p logs/vqgan_imagenet_f16_16384/checkpoints
mkdir -p logs/vqgan_imagenet_f16_16384/configs
wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/configs/model.yaml'
"""

import io

import PIL
import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

from vqgan_models.vqvae import VQModel, load_vqgan_f16_w16384


def load_img_to_device(img_url, device, target_image_size=384):
    resp = requests.get(img_url)
    resp.raise_for_status()
    img = PIL.Image.open(io.BytesIO(resp.content))
    
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
    
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=InterpolationMode.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0).to(device)
    return img * 2. - 1.  # [0,1] to [-1,1]


def tensor_to_pil(chw):
    chw = chw.detach().cpu()
    chw = torch.clamp(chw, -1., 1.)
    chw = (chw + 1.) / 2.  # [-1,1] to [0,1]
    chw = chw.permute(1, 2, 0).numpy()
    chw = (255 * chw).astype(np.uint8)
    chw = Image.fromarray(chw)
    if not chw.mode == 'RGB':
        chw = chw.convert('RGB')
    return chw


def reconstruct_with_vqgan(x_vqgan: torch.Tensor, model16384: VQModel):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    std_z_feat_map, emb_loss, [perplexity, min_encodings, min_encoding_indices] = model16384.encode(x_vqgan)
    model16384.quantize.forward
    std_xrec = model16384.decode(std_z_feat_map)
    
    B = x_vqgan.shape[0]
    xrec = model16384.decode_code(B, min_encoding_indices)
    
    assert torch.allclose(std_xrec, xrec, rtol=1e-5, atol=1e-4)  # todo: remove this
    return std_xrec, xrec


def stack_reconstructions(input, x0, x1, titles=[]):
    assert input.size == x1.size
    w, h = input.size[0], input.size[1]
    img = Image.new("RGB", (3 * w, h))
    img.paste(input, (0, 0))
    img.paste(x0, (1 * w, 0))
    img.paste(x1, (2 * w, 0))
    for i, title in enumerate(titles):
        ImageDraw.Draw(img).text((i * w, 0), f'{title}', (255, 255, 255))  # coordinates, text, color, font
    return img

def main():
    torch.set_grad_enabled(False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model16384 = load_vqgan_f16_w16384('logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt', device)
    inp = load_img_to_device('https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1', device)
    print(f'input is of size: {inp.shape}')
    
    std_xrec, xrec = reconstruct_with_vqgan(inp, model16384)
    print(f'std_xrec: {std_xrec.shape}, xrec: {xrec.shape}')
    img = stack_reconstructions(tensor_to_pil(inp[0]), tensor_to_pil(std_xrec[0]), tensor_to_pil(xrec[0]), titles=['Input', 'VQGAN (f16, 16384) STD', 'VQGAN (f16, 16384) Mine'])
    img


if __name__ == '__main__':
    main()
