# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.base_framework import BaseFramework
from framework.generation_utils import top_k_top_p_filtering
from .PARAMS import SAMPLE, TEMPERATURE


class CoMAE(BaseFramework):
    def __init__(self, encoder=None, decoder=None, toker=None, **kwargs):
        super().__init__(decoder.config)
        self.decoder = decoder
        self.toker = toker
        
        self.pred_loss_weight = kwargs.get('pred_loss_weight', 1.0)
        
        hs = self.decoder.config.n_embd

        self.er_embeddings = nn.Embedding(2, hs)
        self.er_embeddings.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
        self.er_head = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(),)

        self.in_embeddings = nn.Embedding(2, hs)
        self.in_embeddings.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
        self.in_head = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(),)

        self.ex_embeddings = nn.Embedding(2, hs)
        self.ex_embeddings.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
        self.ex_head = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(),)

        self.dialact_dim = 9
        self.dialact_embeddings = nn.Embedding(self.dialact_dim, hs)
        self.dialact_embeddings.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
        self.dialact_head = nn.Sequential(nn.Linear(2 * hs, hs), nn.Tanh(),)

        self.emotion_dim = 10
        self.emotion_embeddings = nn.Embedding(self.emotion_dim, hs)
        self.emotion_embeddings.weight.data.normal_(mean=0.0, std=self.decoder.config.initializer_range)
        self.emotion_head = nn.Sequential(nn.Linear(3 * hs, hs), nn.Tanh(),)
    
    def encode(
        self,
        src_input_ids,
        src_attention_mask,
        src_position_ids,
        src_token_type_ids,
        src_additive_embeds=None,
        encoded_info=None,
    ):
        _, past_key_values, hidden_states = self.decoder(
            input_ids=src_input_ids,
            position_ids=src_position_ids,
            token_type_ids=src_token_type_ids,
            attention_mask=src_attention_mask,
            additive_embeds=src_additive_embeds,
            use_cache=True,
        )
        encoded_info.update({
            'past_key_values': past_key_values,
            'past_attention_mask': src_attention_mask,
        })

        self.predict_mechanism(hidden_states, encoded_info)
        self.predict_dialact(hidden_states, encoded_info)
        self.predict_emotion(hidden_states, encoded_info)

    def predict_mechanism(self, hidden_states, encoded_info):
        tgt_er = encoded_info.get('tgt_er', None)
        tgt_in = encoded_info.get('tgt_in', None)
        tgt_ex = encoded_info.get('tgt_ex', None)
        assert (tgt_er is None) == (tgt_in is None) == (tgt_ex is None)
    
        src_len = encoded_info['src_len']
        idx = torch.arange(0, hidden_states.size(0), dtype=torch.long, device=hidden_states.device)
        last_hidden_states = hidden_states[idx, src_len - 1].contiguous()

        er_logits = F.linear(self.er_head(last_hidden_states), self.er_embeddings.weight)
        in_logits = F.linear(self.in_head(last_hidden_states), self.in_embeddings.weight)
        ex_logits = F.linear(self.ex_head(last_hidden_states), self.ex_embeddings.weight)
    
        if tgt_er is not None:
            loss = F.cross_entropy(er_logits, tgt_er, reduction='mean') + \
                   F.cross_entropy(in_logits, tgt_in, reduction='mean') + \
                   F.cross_entropy(ex_logits, tgt_ex, reduction='mean')
            pred_er = tgt_er
            pred_in = tgt_in
            pred_ex = tgt_ex
        else:
            loss = None
            if SAMPLE:
                pred_er = torch.multinomial(F.softmax(er_logits / TEMPERATURE, dim=-1), num_samples=1).squeeze(-1)
                pred_in = torch.multinomial(F.softmax(in_logits / TEMPERATURE, dim=-1), num_samples=1).squeeze(-1)
                pred_ex = torch.multinomial(F.softmax(ex_logits / TEMPERATURE, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred_er = torch.argmax(er_logits, dim=-1)
                pred_in = torch.argmax(in_logits, dim=-1)
                pred_ex = torch.argmax(ex_logits, dim=-1)
    
        embeds = self.er_embeddings(pred_er) + self.in_embeddings(pred_in) + self.ex_embeddings(pred_ex)
        pred_er_top1 = torch.topk(er_logits, k=1, dim=-1)[1]
        pred_in_top1 = torch.topk(in_logits, k=1, dim=-1)[1]
        pred_ex_top1 = torch.topk(ex_logits, k=1, dim=-1)[1]
    
        encoded_info.update({
            'mechanism_pred_loss': loss,
            'pred_er': pred_er,
            'pred_in': pred_in,
            'pred_ex': pred_ex,
            'pred_er_top1': pred_er_top1,
            'pred_in_top1': pred_in_top1,
            'pred_ex_top1': pred_ex_top1,
            'tgt_mechanism_additive_embeds': embeds,
        })

    def predict_dialact(self, hidden_states, encoded_info):
        tgt_dialact_id = encoded_info.get('tgt_dialact_id', None)
        src_len = encoded_info['src_len']
        idx = torch.arange(0, hidden_states.size(0), dtype=torch.long, device=hidden_states.device)
        last_hidden_states = hidden_states[idx, src_len - 1].contiguous()
        logits = F.linear(self.dialact_head(
            torch.cat([last_hidden_states, encoded_info['tgt_mechanism_additive_embeds']], dim=-1)
        ), self.dialact_embeddings.weight)
    
        if tgt_dialact_id is not None:
            loss = F.cross_entropy(logits, tgt_dialact_id, reduction='mean')
            pred = tgt_dialact_id
        else:
            loss = None
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)
    
        embeds = self.dialact_embeddings(pred)
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        encoded_info.update({
            'dialact_pred_loss': loss,
            'pred_dialact_id': pred,
            'pred_dialact_id_top1': pred_top1,
            'pred_dialact_id_top3': pred_top3,
            'tgt_dialact_additive_embeds': embeds,
        })
    
    def predict_emotion(self, hidden_states, encoded_info):
        tgt_emotion_id = encoded_info.get('tgt_emotion_id', None)
        src_len = encoded_info['src_len']
        idx = torch.arange(0, hidden_states.size(0), dtype=torch.long, device=hidden_states.device)
        last_hidden_states = hidden_states[idx, src_len - 1].contiguous()
        logits = F.linear(self.emotion_head(
            torch.cat([last_hidden_states, encoded_info['tgt_mechanism_additive_embeds'], encoded_info['tgt_emotion_additive_embeds']], dim=-1)
        ), self.emotion_embeddings.weight)
        
        if tgt_emotion_id is not None:
            loss = F.cross_entropy(logits, tgt_emotion_id, reduction='mean')
            pred = tgt_emotion_id
        else:
            loss = None
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)
        
        embeds = self.emotion_embeddings(pred)
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
        
        encoded_info.update({
            'emotion_pred_loss': loss,
            'pred_emotion_id': pred,
            'pred_emotion_id_top1': pred_top1,
            'pred_emotion_id_top3': pred_top3,
            'tgt_emotion_additive_embeds': embeds,
        })
    
    def forward(
        self,
        src_input_ids,
        src_attention_mask,
        src_token_type_ids,
        tgt_input_ids,
        tgt_token_type_id,
        tgt_label_ids,
        pointwise=False,
        **kwargs,
    ):
        encoded_info = kwargs
        assert 'src_dialact_ids' in kwargs and 'tgt_dialact_id' in kwargs
        assert 'src_emotion_ids' in kwargs and 'tgt_emotion_id' in kwargs

        src_additive_embeds = self.dialact_embeddings(kwargs['src_dialact_ids']) + \
                              self.emotion_embeddings(kwargs['src_emotion_ids'])
        src_position_ids = torch.cumsum(src_attention_mask, dim=-1).type_as(src_input_ids) - 1
        self.encode(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            src_position_ids=src_position_ids,
            src_token_type_ids=src_token_type_ids,
            src_additive_embeds=src_additive_embeds,
            encoded_info=encoded_info
        )

        tgt_additive_embeds = encoded_info.pop('tgt_mechanism_additive_embeds') + \
                              encoded_info.pop('tgt_dialact_additive_embeds') + \
                              encoded_info.pop('tgt_emotion_additive_embeds')
        tgt_additive_embeds = tgt_additive_embeds.unsqueeze(1).repeat(1, tgt_input_ids.size(1), 1)

        past_attention_mask = encoded_info.pop('past_attention_mask')
        tgt_attention_mask = torch.cat([past_attention_mask,
                                        past_attention_mask.new_ones((past_attention_mask.size(0), tgt_input_ids.size(1)))], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1)[:, -tgt_input_ids.size(1):].type_as(src_input_ids) - 1
        tgt_token_type_ids = tgt_token_type_id.unsqueeze(-1).repeat(1, tgt_input_ids.size(1))
        outputs = self.decoder(
            input_ids=tgt_input_ids,
            position_ids=tgt_position_ids,
            attention_mask=tgt_attention_mask, # should be concat with src_attention_mask
            token_type_ids=tgt_token_type_ids,
            additive_embeds=tgt_additive_embeds,
            **encoded_info
        )
        lm_logits = outputs[0]

        mechanism_pred_loss = encoded_info['mechanism_pred_loss']
        dialact_pred_loss = encoded_info['dialact_pred_loss']
        emotion_pred_loss = encoded_info['emotion_pred_loss']

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                               ignore_index=-1, reduction='none')
        loss = loss.view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(loss)
        loss_value = torch.sum(loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        
        if not pointwise:
            res = {
                'all': loss_value + self.pred_loss_weight * (mechanism_pred_loss + dialact_pred_loss + emotion_pred_loss),
                'ppl': ppl_value,
                'cm': mechanism_pred_loss,
                'da': dialact_pred_loss,
                'em': emotion_pred_loss,
            }
            return res
        else:
            return loss, label_size
    
    @torch.no_grad()
    def generate(
        self,
        src_input_ids,
        src_attention_mask,
        src_token_type_ids,
        tgt_input_ids,
        tgt_token_type_id,
        **kwargs
    ):
        encoded_info = kwargs
        assert 'src_dialact_ids' in kwargs
        assert 'src_emotion_ids' in kwargs
        assert tgt_input_ids.size(1) == 1

        src_additive_embeds = self.dialact_embeddings(kwargs['src_dialact_ids']) + \
                              self.emotion_embeddings(kwargs['src_emotion_ids'])
        src_position_ids = torch.cumsum(src_attention_mask, dim=-1).type_as(src_input_ids) - 1
        self.encode(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            src_position_ids=src_position_ids,
            src_token_type_ids=src_token_type_ids,
            src_additive_embeds=src_additive_embeds,
            encoded_info=encoded_info
        )

        tgt_additive_embeds = encoded_info.pop('tgt_mechanism_additive_embeds') + \
                              encoded_info.pop('tgt_dialact_additive_embeds') + \
                              encoded_info.pop('tgt_emotion_additive_embeds')
        encoded_info['additive_embeds'] = tgt_additive_embeds.unsqueeze(1)

        past_attention_mask = encoded_info.pop('past_attention_mask')
        tgt_attention_mask = torch.cat([past_attention_mask,
                                        past_attention_mask.new_ones((past_attention_mask.size(0), 1))], dim=-1)
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1)[:, -1:].type_as(src_input_ids) - 1
        tgt_token_type_ids = tgt_token_type_id.unsqueeze(-1)
        
        encoded_info.update({'position_ids': tgt_position_ids})
        encoded_info.update({'token_type_ids': tgt_token_type_ids})

        return encoded_info, super().generate(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, **encoded_info)
