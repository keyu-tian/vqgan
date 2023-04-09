"""
git clone https://github.com/keyu-tian/vqgan.git && cd vqgan/scripts && mkdir -p logs/vqgan_imagenet_f16_16384/checkpoints && mkdir -p logs/vqgan_imagenet_f16_16384/configs && wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' && wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/configs/model.yaml'
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

from .vqgan_models.vqvae import VQModel

# sys.path.append('.')    # to import from the current directory
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_vqgan_ps16_w16384(ckpt_path=None):
    kw = {
        'embed_dim':  256,
        'n_embed':    16384,
        'ddconfig':   {
            'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3,
            'ch':       128, 'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16],
            'dropout':  0.0
        },
        'lossconfig': {
            'target': 'taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False, 'disc_in_channels': 3, 'disc_start': 0, 'disc_weight': 0.75, 'disc_num_layers': 2, 'codebook_weight': 1.0
            }
        }
    }
    
    model = VQModel(**kw).to(DEVICE)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected if not k.startswith('loss.')]
        print(f'missing keys: {missing}')
        print(f'unexpected keys: {unexpected}')
    return model.eval()


def preprocess(img_url, target_image_size=384):
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
    img = torch.unsqueeze(T.ToTensor()(img), 0).to(DEVICE)
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
    old_z_feat_map, emb_loss, [perplexity, min_encodings, min_encoding_indices] = model16384.encode(x_vqgan)
    model16384.quantize.forward
    B = x_vqgan.shape[0]
    z_feat_map = model16384.decode_code(B, min_encoding_indices)
    assert torch.allclose(old_z_feat_map, z_feat_map)  # todo: remove this
    print(f"VQGAN --- {model16384.__class__.__name__}: latent shape: {old_z_feat_map.shape[2:]}")
    
    old_xrec = model16384.decode(old_z_feat_map)
    xrec = model16384.decode(z_feat_map)
    assert torch.allclose(old_xrec, xrec)  # todo: remove this
    return old_xrec


def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
    assert input.size == x1.size == x2.size == x3.size
    w, h = input.size[0], input.size[1]
    img = Image.new("RGB", (5 * w, h))
    img.paste(input, (0, 0))
    img.paste(x0, (1 * w, 0))
    img.paste(x1, (2 * w, 0))
    img.paste(x2, (3 * w, 0))
    img.paste(x3, (4 * w, 0))
    for i, title in enumerate(titles):
        ImageDraw.Draw(img).text((i * w, 0), f'{title}', (255, 255, 255))  # coordinates, text, color, font
    return img


model16384 = load_vqgan_ps16_w16384(ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt")

x_vqgan = preprocess('https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1')
print(f'input is of size: {x_vqgan.shape}')

x1 = reconstruct_with_vqgan(x_vqgan, model16384)
img = stack_reconstructions(tensor_to_pil(x_vqgan[0]), tensor_to_pil(x1[0]), titles=['Input', 'VQGAN (f16, 16384)'])
