# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
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

from typing import Iterable, List, Optional, Tuple
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging
logger = logging.get_logger(__name__)


@torch.no_grad()
def generate(
    self,
    input_ids=None,
    max_length=None,
    min_length=None,
    do_sample=None,
    early_stopping=None,
    num_beams=None,
    temperature=None,
    top_k=None,
    top_p=None,
    repetition_penalty=None,
    bad_words_ids=None,
    bos_token_id=None,
    pad_token_id=None,
    unk_token_id=None,
    eos_token_id=None,
    length_penalty=None,
    no_repeat_ngram_size=None,
    num_return_sequences=None,
    attention_mask=None,
    **model_kwargs
) -> torch.LongTensor:

    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    unk_token_id = unk_token_id if unk_token_id is not None else self.config.unk_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = model_kwargs.pop('batch_size', 1)

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert (bos_token_id is None) or (
        isinstance(bos_token_id, int) and (bos_token_id >= 0)
    ), "`bos_token_id` should be a positive integer."
    assert (unk_token_id is None) or (
        isinstance(unk_token_id, int) and (unk_token_id >= 0)
    ), "`unk_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
        bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"
    
    # not allow to duplicate outputs when greedy decoding
    if not do_sample:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"


    position_ids = model_kwargs.pop('position_ids', None)
    if position_ids is None:
        position_ids = torch.full(
            (batch_size, 1),
            0,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
    elif input_ids is not None:
        assert position_ids.size() == input_ids.size(), 'position_ids have different shape from input_ids'
    
    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
        
    token_type_ids = model_kwargs.pop('token_type_ids', None)
    if token_type_ids is not None:
        if token_type_ids.dim() == 2:
            assert token_type_ids.size() == input_ids.size() or token_type_ids.size(1) == 1, 'token_type_ids have different shape from input_ids'
        else:
            raise ValueError
        
    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # vocab size
    if hasattr(self.config, "vocab_size"):
        vocab_size = self.config.vocab_size
    else:
        raise ValueError("either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined")

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1
    
    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        def reshape(x):
            x_shape = x.size()
            x = x.unsqueeze(1).repeat(1, effective_batch_mult * num_beams, *((1,) * (len(x_shape) - 1)))
            x = x.contiguous().view(-1, *(x_shape[1:]))
            return x

        input_ids = reshape(input_ids)
        position_ids = reshape(position_ids)
        attention_mask = reshape(attention_mask)
        
        if token_type_ids is not None:
            token_type_ids = reshape(token_type_ids)
        # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        
        for key in [
            'inputs_embeds',
            'src_input_ids',
            'additive_embeds',
            'encoder_hidden_states', 'encoder_attention_mask',
            'knl_hidden_states', 'knl_attention_mask'
        ]:
            if key in model_kwargs:
                model_kwargs[key] = reshape(model_kwargs.pop(key))
        
        if 'past_key_values' in model_kwargs:
            past_key_values = []
            for past_key_value in model_kwargs.pop('past_key_values'):
                past_key_values.append(torch.cat([reshape(past_key_value[0]).unsqueeze(0), reshape(past_key_value[1]).unsqueeze(0)], dim=0))
            model_kwargs['past_key_values'] = tuple(past_key_values)
    
    if num_beams > 1:
        output = _generate_beam_search(
            self,
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            model_kwargs=model_kwargs,
        )
    else:
        output = _generate_no_beam_search(
            self,
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            model_kwargs=model_kwargs,
        )

    return output


@torch.no_grad()
def _generate_no_beam_search(
    self,
    input_ids,
    position_ids,
    attention_mask,
    token_type_ids,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    bos_token_id,
    pad_token_id,
    unk_token_id,
    eos_token_id,
    batch_size,
    model_kwargs,
):
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """
    
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    output_ids = input_ids.new_zeros([input_ids.size(0), 0])
    original_input_ids = input_ids

    past_key_values = model_kwargs.pop('past_key_values', None)
    expand_vocab_size = model_kwargs.get('expand_vocab_size', None)
    
    gen_len = 0
    while gen_len < max_length:
        prepared_input_ids = input_ids
        outputs = self.generate_step(
            input_ids=prepared_input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=True,
            **model_kwargs,
        )
        next_token_logits = outputs['lm_logits'][:, -1, :]
        if expand_vocab_size is not None:
            next_token_logits = next_token_logits[:, :-expand_vocab_size]
        
        scores = self.postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=torch.cat([original_input_ids, output_ids], dim=-1) if 'src_input_ids' not in model_kwargs else
                                torch.cat([model_kwargs['src_input_ids'], original_input_ids, output_ids], dim=-1),
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=gen_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
            unk_token_id=unk_token_id,
            bos_token_id=bos_token_id,
        )

        # if model has past, then set the past variable to speed up decoding
        past_key_values = outputs['past_key_values']
        
        if 'updated_kwargs' in outputs:
            model_kwargs.update(outputs.pop('updated_kwargs'))

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + eos_token_id * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = tokens_to_add.unsqueeze(-1)
        output_ids = torch.cat([output_ids, input_ids], dim=-1)

        position_ids = position_ids[:, -1:] + 1
        
        if 'additive_embeds' in model_kwargs:
            model_kwargs['additive_embeds'] = model_kwargs.pop('additive_embeds')[:, -1:]
        
        if attention_mask is not None:
            add_mask = attention_mask.new_ones([attention_mask.shape[0], 1], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, add_mask], dim=1)
        
        gen_len += 1
        unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break
        
    return output_ids


@torch.no_grad()
def _generate_beam_search(
    self,
    input_ids,
    position_ids,
    attention_mask,
    token_type_ids,
    max_length,
    min_length,
    do_sample,
    early_stopping,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    bos_token_id,
    pad_token_id,
    unk_token_id,
    eos_token_id,
    batch_size,
    num_return_sequences,
    length_penalty,
    num_beams,
    model_kwargs,
):
    """Generate sequences for each example with beam search."""
    output_ids = input_ids.new_zeros([input_ids.size(0), 0])
    original_input_ids = input_ids
    
    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
    
    # cache compute states
    past_key_values = model_kwargs.pop('past_key_values', None)
    expand_vocab_size = model_kwargs.get('expand_vocab_size', None)

    # done sentences
    done = [False for _ in range(batch_size)]
    
    gen_len = 0
    while gen_len < max_length:
        prepared_input_ids = input_ids
        outputs = self.generate_step(
            input_ids=prepared_input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=True,
            **model_kwargs,
        )  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs['lm_logits'][:, -1, :] # (batch_size * num_beams, vocab_size)
        if expand_vocab_size is not None: # ensure the final dim has size of vocab_size
            next_token_logits = next_token_logits[:, :-expand_vocab_size]
        vocab_size = next_token_logits.size(-1)

        # if model has past, then set the past variable to speed up decoding
        past_key_values = outputs['past_key_values']

        if 'updated_kwargs' in outputs:
            model_kwargs.update(outputs.pop('updated_kwargs'))

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        
        scores = self.postprocess_next_token_scores(
            scores=scores,
            input_ids=torch.cat([original_input_ids, output_ids], dim=-1)
                        if 'src_input_ids' not in model_kwargs else
                        torch.cat([model_kwargs['src_input_ids'], original_input_ids, output_ids], dim=-1),
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=gen_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
            unk_token_id=unk_token_id,
            bos_token_id=bos_token_id,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # Temperature
            if temperature != 1.0:
                _scores = _scores / temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        output_ids[effective_beam_id].clone(),
                        beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), gen_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = beam_tokens.unsqueeze(1)
        output_ids = output_ids[beam_idx]
        output_ids = torch.cat([output_ids, input_ids], dim=-1)
        if 'additive_embeds' in model_kwargs:
            model_kwargs['additive_embeds'] = model_kwargs.pop('additive_embeds')[beam_idx]
        
        position_ids = position_ids[:, -1].unsqueeze(-1) + 1
        
        if 'additive_embeds' in model_kwargs:
            model_kwargs['additive_embeds'] = model_kwargs.pop('additive_embeds')[:, -1:]
        
        if attention_mask is not None:
            add_mask = attention_mask.new_ones([attention_mask.shape[0], 1], device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, add_mask], dim=1)
        
        gen_len += 1

        # re-order internal states
        if past_key_values is not None:
            past_key_values = self.decoder._reorder_cache(past_key_values, beam_idx)
    
    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx],
                beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # prepare for adding eos
    sent_max_len = min(sent_lengths.max().item() + 1, max_length)
    decoded = input_ids.new(output_batch_size, sent_max_len)
    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`pad_token_id` has to be defined"
        decoded.fill_(pad_token_id)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, hypo in enumerate(best):
        decoded[i, :sent_lengths[i]] = hypo
        if sent_lengths[i] < max_length:
            decoded[i, sent_lengths[i]] = eos_token_id

    return decoded


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -1e5,
    min_tokens_to_keep: int = 1
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
