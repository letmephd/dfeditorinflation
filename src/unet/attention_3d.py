# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.cross_attention import CrossAttention
from diffusers.models.attention import FeedForward, AdaLayerNorm
# import jax
# import jax.numpy as jnp

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)
        attention_map_list = []
        # Blocks
        # print("check dimension") # hidden_states torch.Size([16, 4096, 320])
        # from IPython import embed;embed()
        for block in self.transformer_blocks:
            hidden_states,attention_map = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )
            attention_map_list.append(attention_map)

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=(output,attention_map_list))


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        # self.attn1 = MyCrossAttention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = MyCrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            # self.attn2 = CrossAttention(
            #     query_dim=dim,
            #     cross_attention_dim=cross_attention_dim,
            #     heads=num_attention_heads,
            #     dim_head=attention_head_dim,
            #     dropout=dropout,
            #     bias=attention_bias,
            #     upcast_attention=upcast_attention,
            # )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        # self.attn_temp = CrossAttention(
        #     query_dim=dim,
        #     heads=num_attention_heads,
        #     dim_head=attention_head_dim,
        #     dropout=dropout,
        #     bias=attention_bias,
        #     upcast_attention=upcast_attention,
        # )
        # nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        # self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool,useless:None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        
        
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        ) # [2, 9216, 320]

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            # norm_hidden_states: 2, 4096, 320
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states_, attention_map = self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
            hidden_states = (hidden_states_+ hidden_states)
        else:
            attention_map = None
            # hidden_states = (
            #     self.attn2(
            #         norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            #     )
            #     + hidden_states
            # )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        # norm_hidden_states = (
        #     self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        # )
        # hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states,attention_map


class SparseCausalAttention(CrossAttention):
    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    
    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
    # def reshape_heads_to_batch_dim(self, tensor):
    #     batch_size, seq_len, dim = tensor.shape
    #     head_size = self.heads
    #     tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    #     tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    #     tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
    #     return tensor
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        # print("SCA")
        # from IPython import embed;embed()
        encoder_hidden_states = encoder_hidden_states # 2, 4096, 320

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # # 2, 4096, 320
        dim = query.shape[-1] #320
        # query = self.reshape_heads_to_batch_dim(query)
        query = self.reshape_heads_to_batch_dim(query) # [16, 4096, 40]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        
        
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states #[2, 4096, 320])
        key = self.to_k(encoder_hidden_states) # [2, 4096, 320]
        value = self.to_v(encoder_hidden_states) # [2, 4096, 320]

        former_frame_index = torch.arange(video_length) - 1 # tensor([-1])
        former_frame_index[0] = 0
        # print("checkselfattention")
        # from IPython import embed;embed()

        # key 32, 4096, 320
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length) # 2, 1, 4096, 320] torch.Size([2, 16, 4096, 320])
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2) # [2, 1, 8192, 320] # torch.Size([2, 16, 8192, 320])
        key = rearrange(key, "b f d c -> (b f) d c") # [2, 8192, 320] #torch.Size([32, 8192, 320])
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length) # 2, 1, 4096, 320
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2) # 2, 1, 8192, 320
        value = rearrange(value, "b f d c -> (b f) d c") #  2, 8192, 320 

        # key = self.reshape_heads_to_batch_dim(key)
        # value = self.reshape_heads_to_batch_dim(value)
        
        key = self.reshape_heads_to_batch_dim(key) # 16, 8192, 40 ,256, 8192, 40
        value = self.reshape_heads_to_batch_dim(value) # 16, 8192, 40

        if attention_mask is not None: #False
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value, attention_mask)
        #     else:
        #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        # print("attention")
        # from IPython import embed;embed()
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value, attention_mask)
        #     else:
        #         hidden_states = self._sliced_attention(
        #             query, key, value, hidden_states.shape[1], dim, attention_mask
        #         )
        if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = self._attention(query, key, value, attention_mask)
        # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #     hidden_states = self._attention(query, key, value, attention_mask)
        # else:
        #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        # linear proj
        # from IPython import embed;embed()
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

class MyCrossAttention(CrossAttention):
    
    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states,attention_probs
    
    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    
    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
    # def reshape_heads_to_batch_dim(self, tensor):
    #     batch_size, seq_len, dim = tensor.shape
    #     head_size = self.heads
    #     tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    #     tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    #     tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
    #     return tensor
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape
        # print("SCA")
        # from IPython import embed;embed()
        encoder_hidden_states = encoder_hidden_states # 2, 4096, 320

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # # 2, 4096, 320
        dim = query.shape[-1] #320
        # query = self.reshape_heads_to_batch_dim(query)
        query = self.reshape_heads_to_batch_dim(query) # [16, 4096, 40]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        # print("checkcrossattention")
        # from IPython import embed;embed()

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # key = self.reshape_heads_to_batch_dim(key)
        # value = self.reshape_heads_to_batch_dim(value)
        
        # key = self.reshape_heads_to_batch_dim(key) # 16, 8192, 40
        # value = self.reshape_heads_to_batch_dim(value) # 16, 8192, 40

        if attention_mask is not None: #False
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        #             query, key, value, hidden_states.shape[1], dim, attention_mask
        #         )
        if self._use_memory_efficient_attention_xformers and query.shape[-2] > 32 ** 2:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
                attenton_map=None
        else:
            hidden_states,attenton_map = self._attention(query, key, value, attention_mask)
        # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #     hidden_states = self._attention(query, key, value, attention_mask)
        # else:
        #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        # linear proj
        # from IPython import embed;embed()
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states,attenton_map
