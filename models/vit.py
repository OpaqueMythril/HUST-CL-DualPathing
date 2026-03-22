import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import adapt_input_conv


# --- 基础组件：MLP ---
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --- 基础组件：Attention (支持 Prompt 注入) ---
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False, prompt=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 核心：处理 Prompt 注入 ---
        if prompt is not None:
            pk, pv = prompt  # 期待格式为 [Batch, Lp, Dim]
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# --- 基础组件：Transformer Block ---
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False, prompt=None):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- 主类：Vision Transformer ---
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 ckpt_layer=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
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

    # --- 新增：专门用于获取 768 维特征的接口 ---
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # 返回 [Batch, Tokens, 768]

    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, task_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        for i, blk in enumerate(self.blocks):
            p_list = None
            if prompt is not None:
                # 兼容 Tensor 列表（双路生成方案）和 原版对象（CODA方案）
                if isinstance(prompt, (list, tuple)):
                    p_list = prompt
                else:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True, task_id=task_id)
                        prompt_loss += loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False, task_id=task_id)

            x = blk(x, register_blk == i, prompt=p_list)

        x = self.norm(x)
        return x, prompt_loss


# --- 权重加载逻辑 ---
@torch.no_grad()
def _load_weights(model, checkpoint_path, prefix=''):
    import numpy as np
    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)

    @torch.no_grad()
    def _load_weights(model, checkpoint_path, prefix=''):
        """ 从官方 .npz 检查点加载权重到 PyTorch 模型 """
        import numpy as np

        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        # 1. 加载文件
        w = np.load(checkpoint_path)
        if not prefix and 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'

        # 2. 映射 Patch Embedding
        if hasattr(model.patch_embed, 'proj'):
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1],
                _n2p(w[f'{prefix}embedding/kernel'])
            )
            model.patch_embed.proj.weight.copy_(embed_conv_w)
            model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

        # 3. 映射 CLS Token 和 Position Embedding
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
        if pos_embed_w.shape != model.pos_embed.shape:
            # 如果尺寸不匹配，此处通常需要调用 interpolate_pos_embed 辅助函数
            model.pos_embed.copy_(pos_embed_w[:, :model.pos_embed.shape[1], :])
        else:
            model.pos_embed.copy_(pos_embed_w)

        # 4. 映射最后的一层 LayerNorm
        model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

        # 5. 循环映射 12 个 Transformer Blocks
        for i, block in enumerate(model.blocks):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'

            # Norm1
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))

            # Attention QKV (JAX 分开存储，PyTorch 拼接)
            q_w = _n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T
            k_w = _n2p(w[f'{mha_prefix}key/kernel'], t=False).flatten(1).T
            v_w = _n2p(w[f'{mha_prefix}value/kernel'], t=False).flatten(1).T
            block.attn.qkv.weight.copy_(torch.cat([q_w, k_w, v_w]))

            q_b = _n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1)
            k_b = _n2p(w[f'{mha_prefix}key/bias'], t=False).reshape(-1)
            v_b = _n2p(w[f'{mha_prefix}value/bias'], t=False).reshape(-1)
            block.attn.qkv.bias.copy_(torch.cat([q_b, k_b, v_b]))

            # Attention Proj
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))

            # Norm2
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            # MLP
            block.mlp.fc1.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_0/kernel']))
            block.mlp.fc1.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_0/bias']))
            block.mlp.fc2.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_1/kernel']))
            block.mlp.fc2.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_1/bias']))