# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import pdb

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import sobel
import torch.nn.functional as F

from vision_transformer import PatchEmbed, Block, CBlock
from util.pos_embed import get_2d_sincos_pos_embed


class DMaskedAutoencoderConvViT(nn.Module):
    """ Deblurring Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=True, qk_scale=None,
                norm_layer=norm_layer)
            for i in range(depth[2])])
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      (patch_size[0] * patch_size[1] * patch_size[2]) ** 2 * in_chans,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed3.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed3.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
        #        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def fused_masking_ids(self, x, mask_true, mask_ratio, alpha=0.5):
        if mask_true.dim() == 3:
            mask_true = mask_true.unsqueeze(1)  # [B,1,H,W]
        elif mask_true.shape[1] == 3:
            mask_true = mask_true.mean(dim=1, keepdim=True)  # [B,1,H,W]
        B, _, H, W = x.shape
        L = self.patch_embed3.num_patches
        len_keep = int(L * (1 - mask_ratio))
        h = w = int(L ** 0.5)

        patch = int((H * W // L) ** 0.5)
        x_patches = F.unfold(x, kernel_size=patch, stride=patch)  # [B, 3*patch*patch, L]
        x_patches = x_patches.transpose(1, 2)  # [B, L, 3*patch*patch]

        texture_scores = x_patches.var(dim=-1)  # [B, L]

        # ✅ 计算梯度 (边缘强度)
        if mask_true.dim() == 3:
            mask_true = mask_true.unsqueeze(1)

        # Sobel 卷积核
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)

        gx = F.conv2d(mask_true, sobel_x, padding=1)
        gy = F.conv2d(mask_true, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # [B,1,H,W]

        # ✅ 根据 patch 尺寸进行平均池化
        H, W = mask_true.shape[-2:]
        patch = int((H * W // L) ** 0.5)
        edge_scores = F.avg_pool2d(grad_mag, kernel_size=patch, stride=patch).flatten(1)  # [B,L]

        # ✅ 融合纹理和边缘得分
        scores = alpha * texture_scores + (1 - alpha) * edge_scores

        ids_shuffle = torch.argsort(scores, dim=1, descending=False)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_true,mask_ratio,):
        # embed patches
        ids_keep, mask, ids_restore = self.fused_masking_ids(x, mask_true,mask_ratio)
        # 新增的多尺度mask
        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(
            0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(0,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      2,
                                                                                                                      4).reshape(
            x.shape[0], 28, 28).unsqueeze(1)
        # 每一阶段都输出中间特征图s tage1_embed
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        stage1_embed = torch.gather(stage1_embed, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(stage2_embed, dim=1,
                                    index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)
        # 多层特征融合
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean()  # reconstruct and denoise all pixels
        #   loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, imgs_orig, mask_true,mask_ratio=0.75,):
        latent, mask, ids_restore = self.forward_encoder(imgs,mask_true,mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs_orig, pred, mask)
        return loss, pred, mask


def dconvmae_convvit_base_patch16_dec512d8b(**kwargs):
    model = DMaskedAutoencoderConvViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dconvmae_convvit_large_patch16_dec512d8b(**kwargs):
    model = DMaskedAutoencoderConvViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[384, 768, 1024], depth=[2, 2, 23], num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[8, 8, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dconvmae_convvit_huge_patch16_dec512d8b(**kwargs):
    model = DMaskedAutoencoderConvViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[768, 1024, 1280], depth=[2, 2, 31], num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=[8, 8, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
dconvmae_convvit_base_patch16 = dconvmae_convvit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
dconvmae_convvit_large_patch16 = dconvmae_convvit_large_patch16_dec512d8b
dconvmae_convvit_huge_patch16 = dconvmae_convvit_huge_patch16_dec512d8b