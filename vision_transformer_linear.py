import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp


def requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


class LinearBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=197):
        super().__init__()

        # First stage
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)

        # Second stage
        self.mlp2 = Mlp(in_features=num_tokens, hidden_features=int(
            num_tokens * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(num_tokens)

        # Dropout (or a variant)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp1(self.norm1(x)))
        x = x.transpose(-2, -1)
        x = x + self.drop_path(self.mlp2(self.norm2(x)))
        x = x.transpose(-2, -1)
        return x


class PatchEmbed(nn.Module):
    """ Wraps a convolution """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    """ Learned positional encoding with dynamic interpolation at runtime """

    def __init__(self, height, width, embed_dim):
        super().__init__()
        self.height = height
        self.width = width
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, height, width))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        if H == self.height and W == self.width:
            pos_embed = self.pos_embed
        else:
            pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        return self.cls_pos_embed, pos_embed


class LinearVisionTransformer(nn.Module):
    """
    Basically the same as the standard Vision Transformer, but with support for resizable 
    or sinusoidal positional embeddings. 
    """

    def __init__(self, *, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 positional_encoding='learned', learned_positional_encoding_size=(14, 14), block_cls=LinearBlock):
        super().__init__()

        # Config
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding
        if positional_encoding == 'learned':
            height, width = self.learned_positional_encoding_size = learned_positional_encoding_size
            self.pos_encoding = LearnedPositionalEncoding(height, width, embed_dim)
        else:
            raise NotImplementedError('Unsupposed positional encoding')
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            block_cls(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_tokens=1 + (224 // patch_size)**2)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Init
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        # Patch embedding
        B, C, H, W = x.shape  # B x C x H x W
        x = self.patch_embed(x)  # B x E x H//p x W//p

        # Positional encoding
        # NOTE: cls_pos_embed for compatibility with pretrained models
        cls_pos_embed, pos_embed = self.pos_encoding(x)

        # Flatten image, append class token, add positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x.flatten(2).transpose(1, 2)  # flatten
        x = torch.cat((cls_tokens, x), dim=1)  # class token
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # flatten
        pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)  # class pos emb
        x = x + pos_embed
        x = self.pos_drop(x)

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        # Final layernorm
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def linear_tiny(pretrained=False, **kwargs):
    model = LinearVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def linear_base(pretrained=False, **kwargs):
    model = LinearVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def linear_large(pretrained=False, **kwargs):
    model = LinearVisionTransformer(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':

    # Test
    x = torch.randn(2, 3, 224, 224)
    m = linear_tiny()
    out = m(x)
    print('-----')
    print(f'num params: {sum(p.numel() for p in m.parameters())}')
    print(out.shape)
    loss = out.sum()
    loss.backward()
    print('Single iteration completed successfully')
