from typing import Optional, Sequence

import torch
import torch.nn as nn

from dinov2.models.vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)


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
            torch.tensor([self.mri_sequences_default.index(seq) for seq in self.mri_sequences]),
            persistent=False,
        )

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
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
        mri_seq_embed_idc = self.mri_seq_idx_map[torch.arange(nc, device=self.mri_seq_idx_map.device)]

        # create embeddings for the MRI sequences
        mri_seq_embed = self.mri_seq_embed(mri_seq_embed_idc).to(patch_tokens.dtype)
        mri_seq_embed = nn.functional.normalize(mri_seq_embed, p=2, dim=-1) * self.mri_seq_embed_max_norm

        # reshape the MRI sequence embeddings to match the shape of the patches
        mri_seq_embed = mri_seq_embed[None, :, None].expand(B, -1, patch_tokens.shape[2], -1)

        # add the MRI sequence embedding to the patch tokens
        patch_tokens = patch_tokens + mri_seq_embed

        # concatenate the individually embedded channels into a single sequence
        patch_tokens = patch_tokens.view(B, -1, self.embed_dim)

        # only use one cls token per image
        cls_tokens = cls_tokens[:, 0]

        # concatenate the cls token with the patch tokens
        x = torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)

        return x


def glioma_vit_giant2(**kwargs):
    return vit_giant2(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_small(**kwargs):
    return vit_small(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_base(**kwargs):
    return vit_base(vit_cls_=GliomaDinoViT, **kwargs)


def glioma_vit_large(**kwargs):
    return vit_large(vit_cls_=GliomaDinoViT, **kwargs)
