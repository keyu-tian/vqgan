import importlib

import numpy as np
import torch
import torch.nn as nn

from .ddpm import Decoder, Encoder


# def get_obj_from_str(string, reload=False):
#     module, cls = string.rsplit(".", 1)
#     if reload:
#         module_imp = importlib.import_module(module)
#         importlib.reload(module_imp)
#     return getattr(importlib.import_module(module, package=None), cls)
#
#
# def instantiate_from_config(config):
#     if not "target" in config:
#         raise KeyError("Expected key `target` to instantiate.")
#     return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e
        
        self.sane_index_shape = sane_index_shape
    
    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)
    
    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # todo: replace einops with permute and reshape
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # z_flattened = z.view(-1, self.e_dim)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        # todo: replace einops with matmul
        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        #     torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * z_flattened @ self.embedding.weight.T
        )
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        
        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # reshape back to match original input shape
        # todo: replace einops with permute and reshape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten
        
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
        
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    
    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again
        
        # get quantized latent vectors
        z_q = self.embedding(indices)
        
        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q


class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(n_embed, embed_dim, beta=0.25,
                                         remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        self.quantize.forward
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant_bchw):
        quant_bchw = self.post_quant_conv(quant_bchw)
        dec = self.decoder(quant_bchw)
        return dec
    
    def decode_code(self, B, code_b):
        quant_nc = self.quantize.embedding(code_b)
        
        BHW, C = quant_nc.shape
        HW = BHW // B
        H = W = round(np.sqrt(HW))
        quant_bchw = quant_nc.view(B, H, W, C).permute(0, 3, 1, 2)   # NC => BHWC => BCHW
        
        quant_bchw = self.post_quant_conv(quant_bchw) # todo: self.quantize.embedding(code_b)'s shape could be wrong
        dec = self.decode(quant_bchw)
        return dec
    
    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
    #                               list(self.decoder.parameters()) +
    #                               list(self.quantize.parameters()) +
    #                               list(self.quant_conv.parameters()) +
    #                               list(self.post_quant_conv.parameters()),
    #                               lr=lr, betas=(0.5, 0.9))
    #     opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
    #                                 lr=lr, betas=(0.5, 0.9))
    #     return [opt_ae, opt_disc], []
