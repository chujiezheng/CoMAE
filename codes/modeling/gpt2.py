# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Context-Knowledge-aware Transformer Decoder model."""

import warnings
import copy
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from configuration import GPT2Config
from transformers.utils import logging
from modeling.base_decoder import BaseDecoder
from modeling.sinusoidal_positional_embeddings import SinusoidalPositionEmbeddings

logger = logging.get_logger(__name__)



class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """
    
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(nf), requires_grad=True)
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False, ny=None):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            if ny is None:
                ny = nx
            self.c_attn = Conv1D(2 * n_state, ny)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False,
              is_encoder=False):
        
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention and not is_encoder:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = torch.softmax(w, dim=-1) # bs, num_heads, q_len, k_len
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value,
                                  attention_mask=attention_mask,
                                  head_mask=head_mask,
                                  output_attentions=output_attentions,)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)
        
        if hasattr(config, 'n_encoder_embd'):
            n_encoder_embd = config.n_encoder_embd
        else:
            n_encoder_embd = config.n_embd
        if hasattr(config, 'add_cross_attention') and config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True, ny=n_encoder_embd)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if hasattr(config, 'add_knl_attention') and config.add_knl_attention:
            self.knlattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True, ny=n_encoder_embd)
            self.ln_knl_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        knl_hidden_states=None,
        knl_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        self_attn_input = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            self_attn_input,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = [attn_outputs[1]]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_input = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                cross_attn_input,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            
        if knl_hidden_states is not None:
            # add one self-attention block for knl-attention
            assert hasattr(
                self, "knlattention"
            ), f"If `knl_hidden_states` are passed, {self} has to be instantiated with knl-attention layers by setting `config.add_knl_attention=True`"
            knl_attn_input = self.ln_knl_attn(hidden_states)
            
            assert knl_hidden_states.dim() == 3
            knl_attn_outputs = self.knlattention(
                knl_attn_input,
                attention_mask=attention_mask,
                encoder_hidden_states=knl_hidden_states,
                encoder_attention_mask=knl_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = knl_attn_outputs[0]
            
            # residual connection
            hidden_states = hidden_states + attn_output

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (cross_attentions, knl_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2DecoderModel(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
    
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        if hasattr(config, 'static_position_embeddings') and config.static_position_embeddings:
            self.wpe = SinusoidalPositionEmbeddings(config.n_positions, config.n_embd)
        else:
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        self.add_type_vocab = False
        if hasattr(config, 'add_type_vocab') and config.add_type_vocab:
            assert hasattr(config, 'type_vocab_size')
            logger.info('add DIY type vocab and embeddings!')
            self.add_type_vocab = True
            self.tte = nn.Embedding(config.type_vocab_size, config.n_embd)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        additive_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        knl_hidden_states=None,
        knl_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            assert input_ids.dim() == 2
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is not None:
            assert position_ids.size(1) == input_ids.size(1)

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = self.invert_attention_mask(attention_mask)
        
        if encoder_hidden_states is not None:
            assert encoder_attention_mask is not None
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        if knl_hidden_states is not None:
            assert knl_attention_mask is not None
            knl_attention_mask = self.invert_attention_mask(knl_attention_mask)
        
        position_embeds = self.wpe(position_ids)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        if token_type_ids is not None:
            assert token_type_ids.dim() == inputs_embeds.dim() - 1
            assert token_type_ids.size() == inputs_embeds.size()[:-1] or token_type_ids.size(1) == 1
            if self.add_type_vocab:
                token_type_embeds = self.tte(token_type_ids)
            else:
                token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0.
        
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        
        if additive_embeds is not None:
            #if all(e is None for e in past_key_values):
            #    assert additive_embeds.size() == inputs_embeds.size()
            #else:
            #    additive_embeds = additive_embeds[:, -inputs_embeds.size(1):]
            assert additive_embeds.size() == inputs_embeds.size()
            hidden_states = hidden_states + additive_embeds
        
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                knl_hidden_states=knl_hidden_states,
                knl_attention_mask=knl_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions and len(outputs) > 2:
                all_attentions = all_attentions + (outputs[2],)
        
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)


class MyGPT2Decoder(GPT2PreTrainedModel, BaseDecoder):
    def __init__(self, config: GPT2Config):
        config.tie_word_embeddings = True
        super().__init__(config)
        
        self.transformer = GPT2DecoderModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward_lm_head(
        self,
        hidden_states,
    ):
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
    
    def get_word_embeddings(self):
        return self.transformer.wte

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        additive_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        knl_hidden_states=None,
        knl_attention_mask=None,
        use_cache=False,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            additive_embeds=additive_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            knl_hidden_states=knl_hidden_states,
            knl_attention_mask=knl_attention_mask,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.forward_lm_head(hidden_states)
        
        if use_cache:
            presents = transformer_outputs[1]
        else:
            presents = None
        
        return lm_logits, presents, hidden_states,
