# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from dinov2.layers import MemEffAttention, Mlp, PatchEmbed, SwiGLUFFNFused
from dinov2.layers import NestedTensorBlock as Block

logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, ret_feat_dict=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if ret_feat_dict:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


class GliomaDinoViT(DinoVisionTransformer):
    # embedding to indicate the MRI sequence for each token (patch)
    mri_sequences_default = ["t1", "t1c", "t2", "flair"]

    def __init__(
        self,
        mri_sequences: Sequence[str] = None,
        use_mri_seq_embed=True,
        img_wise_pos_embed=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_wise_pos_embed = img_wise_pos_embed
        self.mri_seq_embed_max_norm = 1.0 if use_mri_seq_embed else 0.0
        self.mri_seq_embed = nn.Embedding(
            4,
            self.embed_dim,
        )
        self.mri_sequences = mri_sequences or self.mri_sequences_default
        self.register_buffer(
            "mri_seq_idx_map",
            torch.tensor(
                [self.mri_sequences_default.index(seq) for seq in self.mri_sequences]
            ),
            persistent=False,
        )

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        B, nc, w, h = x.shape

        if not self.img_wise_pos_embed:
            # stack channels along spatial dimensions
            x = x.view(B, 1, nc * w, h)
        else:
            # flatten channels into the batch dimension (ours)
            x = x.view(B * nc, 1, w, h)

        # make image rgb again
        x = x.repeat(1, 3, 1, 1)

        if masks is not None:
            n_tokens = x.shape[-1] // self.patch_size
            if not self.img_wise_pos_embed:
                if masks.numel() == (B * nc * n_tokens**2):
                    # channel-wise masking (masks have been generated for each image)
                    masks = masks.view(B, nc, n_tokens, n_tokens)
                else:  # patch-based masking (ibot)
                    masks = masks.view(B, 1, n_tokens, n_tokens)
                    # repeat masks for each channel along spatial dimensions
                    masks = masks.repeat(1, nc, 1, 1)

                # flatten the masks into the spatial dimension
                masks = masks.flatten(start_dim=1)

            else:  # ours
                if masks.size(1) // (n_tokens**2) != nc:  # no channel-wise masking
                    # masks is B x n_tokens if not channel-wise masking
                    # we need to expand it to B x n_channels x n_tokens
                    masks = masks.unsqueeze(1).expand(B, nc, -1)
                    # flatten the masks
                    masks = masks.flatten(0, 1)
                else:  # channel-wise masking
                    # masks have already been prepared for channel-wise masking (B,nc,n_tokens)
                    # reshape masks to (B*nc, n_tokens)
                    masks = masks.view(B * nc, -1)

        # call the original prepare_tokens_with_masks
        x = super().prepare_tokens_with_masks(x, masks)

        if self.img_wise_pos_embed:
            # reshape MRI sequences into channels
            x = x.view(B, nc, -1, self.embed_dim)

            # each channel has a cls token
            cls_tokens = x[:, :, 0]
            patch_tokens = x[:, :, 1:]

        else:
            # reshape MRI sequences into channels
            x = x.view(B, -1, self.embed_dim)
            cls_tokens = x[:, 0]
            patch_tokens = x[:, 1:]
            cls_tokens = cls_tokens.unsqueeze(1)
            patch_tokens = patch_tokens.view(B, nc, -1, self.embed_dim)

        # map given sequences to default indices for embedding
        mri_seq_embed_idc = self.mri_seq_idx_map[
            torch.arange(nc, device=self.mri_seq_idx_map.device)
        ]

        # create embeddings for the MRI sequences
        mri_seq_embed = self.mri_seq_embed(mri_seq_embed_idc).to(patch_tokens.dtype)
        mri_seq_embed = (
            nn.functional.normalize(mri_seq_embed, p=2, dim=-1)
            * self.mri_seq_embed_max_norm
        )

        # reshape the MRI sequence embeddings to match the shape of the patches
        mri_seq_embed = mri_seq_embed[None, :, None].expand(
            B, -1, patch_tokens.shape[2], -1
        )

        # add the MRI sequence embedding to the patch tokens
        patch_tokens = patch_tokens + mri_seq_embed

        # concatenate the individually embedded channels into a single sequence
        patch_tokens = patch_tokens.view(B, -1, self.embed_dim)

        # only use one cls token per image
        cls_tokens = cls_tokens[:, 0]

        # concatenate the cls token with the patch tokens
        x = torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)

        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def glioma_vit_small(**kwargs):
    return vit_small(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_base(**kwargs):
    return vit_base(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_large(**kwargs):
    return vit_large(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_giant2(**kwargs):
    return vit_giant2(vit_cls_=GliomaDinoViT, **kwargs)


def vit_small(patch_size=16, vis_cls_=DinoVisionTransformer, **kwargs):
    model = vis_cls_(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, vit_cls_=DinoVisionTransformer, **kwargs):
    model = vit_cls_(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, vit_cls_=DinoVisionTransformer, **kwargs):
    model = vit_cls_(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, vit_cls_=DinoVisionTransformer, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = vit_cls_(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model
