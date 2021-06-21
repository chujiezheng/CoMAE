# coding=utf-8

import json
import tqdm
import torch
import random
from typing import List
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputter.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH


GOLDEN_CM = GOLDEN_TRUTH
GOLDEN_DA = GOLDEN_TRUTH
GOLDEN_EM = GOLDEN_TRUTH


class CoMAEInputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features
        
        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader
        
        # valid
        self.valid_dataloader = DynamicBatchingLoader
        
        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(
        self,
        src_input_ids, src_token_type_ids, src_dialact_ids, src_emotion_ids,
        tgt_input_ids, tgt_token_type_id, tgt_label_ids, tgt_dialact_id, tgt_emotion_id,
        tgt_er, tgt_in, tgt_ex,
    ):
        self.src_input_ids = src_input_ids
        self.src_token_type_ids = src_token_type_ids
        self.src_len = len(src_input_ids)
        self.src_dialact_ids = src_dialact_ids
        self.src_emotion_ids = src_emotion_ids
        
        self.tgt_input_ids = tgt_input_ids
        self.tgt_token_type_id = tgt_token_type_id
        self.tgt_len = len(tgt_input_ids)
        self.tgt_label_ids = tgt_label_ids
        self.tgt_dialact_id = tgt_dialact_id
        self.tgt_emotion_id = tgt_emotion_id
        
        self.tgt_er = tgt_er
        self.tgt_in = tgt_in
        self.tgt_ex = tgt_ex

        self.input_len = self.src_len + self.tgt_len


def featurize(
    eos,
    context, max_src_len, segment_ids, src_dialact_ids, src_emotion_ids,
    response, max_tgt_len, tgt_segment_id, tgt_dialact_id, tgt_emotion_id,
    tgt_er, tgt_in, tgt_ex,
):
    post = context[0]
    comments = [[eos] + e for e in context[1:]]
    context = [post] + comments
    assert len(context) == len(segment_ids)
    src_dialact_ids = [[e] * len(text) for e, text in zip(src_dialact_ids, context)]
    src_emotion_ids = [[e] * len(text) for e, text in zip(src_emotion_ids, context)]
    src_token_type_ids = [[e] * len(text) for e, text in zip(segment_ids, context)]
    
    src_input_ids = sum(context, [])[-max_src_len:]
    src_dialact_ids = sum(src_dialact_ids, [])[-max_src_len:]
    src_emotion_ids = sum(src_emotion_ids, [])[-max_src_len:]
    src_token_type_ids = sum(src_token_type_ids, [])[-max_src_len:]
    assert len(src_input_ids) == len(src_dialact_ids) == len(src_emotion_ids) == len(src_token_type_ids)

    tgt_label_ids = (response + [eos])[:max_tgt_len]
    tgt_input_ids = [eos] + tgt_label_ids[:-1]
    tgt_token_type_id = tgt_segment_id
    assert len(tgt_input_ids) == len(tgt_label_ids)
    
    return InputFeatures(
        src_input_ids, src_token_type_ids, src_dialact_ids, src_emotion_ids,
        tgt_input_ids, tgt_token_type_id, tgt_label_ids, tgt_dialact_id, tgt_emotion_id,
        tgt_er, tgt_in, tgt_ex,
    )


def convert_data_to_inputs(data, toker, **kwargs):
    dialog = data['dialog']
    assert 'max_src_turn' in kwargs, 'you should give max_src_turn'
    max_src_turn = kwargs.get('max_src_turn')
    
    sep = toker.sep_token_id
    if sep is None:
        sep = toker.eos_token_id
        assert sep is not None
    
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(_norm(x)))
    
    inputs = []
    context = []
    dialact_ids = []
    emotion_ids = []
    segment_ids = []
    
    for i in range(len(dialog)):
        text = process(dialog[i]['text'])
        dialact_id = dialog[i]['dialact']
        emotion_id = dialog[i]['emotion']
        segment_id = dialog[i]['segment_id']
        assert segment_id in [0, 1]
        
        if i > 0 and dialog[i]['speaker'] == 'sys':
            res = {
                'context': context.copy(),
                'dialact_ids': dialact_ids.copy(),
                'emotion_ids': emotion_ids.copy(),
                'segment_ids': segment_ids.copy(),
                
                'response': text,
                'tgt_dialact_id': dialact_id,
                'tgt_emotion_id': emotion_id,
                'tgt_segment_id': segment_id,
                
                'tgt_er': dialog[i]['er'],
                'tgt_in': dialog[i]['in'],
                'tgt_ex': dialog[i]['ex'],
            }
            
            inputs.append(res)
        
        context = (context + [text])[-max_src_turn:]
        dialact_ids = (dialact_ids + [dialact_id])[-max_src_turn:]
        emotion_ids = (emotion_ids + [emotion_id])[-max_src_turn:]
        segment_ids = (segment_ids + [segment_id])[-max_src_turn:]
    
    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []
    
    max_src_len = kwargs.get('max_src_len')
    max_tgt_len = kwargs.get('max_tgt_len')
    eos = toker.eos_token_id
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            eos,
            ipt['context'], max_src_len, ipt['segment_ids'], ipt['dialact_ids'], ipt['emotion_ids'],
            ipt['response'], max_tgt_len, ipt['tgt_segment_id'], ipt['tgt_dialact_id'], ipt['tgt_emotion_id'],
            ipt['tgt_er'], ipt['tgt_in'], ipt['tgt_ex'],
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        
        src_input_ids = pad_sequence([torch.tensor(f.src_input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        src_attention_mask = pad_sequence([torch.tensor([1.] * f.src_len, dtype=torch.float) for f in features],
                          batch_first=True, padding_value=0.)
        src_token_type_ids = pad_sequence([torch.tensor(f.src_token_type_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=0)
        src_dialact_ids = pad_sequence([torch.tensor(f.src_dialact_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=0)
        src_emotion_ids = pad_sequence([torch.tensor(f.src_emotion_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=0)
        src_len = torch.tensor([f.src_len for f in features], dtype=torch.long)
        
        tgt_input_ids = pad_sequence([torch.tensor(f.tgt_input_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=pad)
        tgt_label_ids = pad_sequence([torch.tensor(f.tgt_label_ids, dtype=torch.long) for f in features],
                          batch_first=True, padding_value=-1)
        tgt_token_type_id = torch.tensor([f.tgt_token_type_id for f in features], dtype=torch.long)
        tgt_dialact_id = torch.tensor([f.tgt_dialact_id for f in features], dtype=torch.long)
        tgt_emotion_id = torch.tensor([f.tgt_emotion_id for f in features], dtype=torch.long)
        tgt_len = torch.tensor([f.tgt_len for f in features], dtype=torch.long)
        
        tgt_er = torch.tensor([f.tgt_er for f in features], dtype=torch.long)
        tgt_in = torch.tensor([f.tgt_in for f in features], dtype=torch.long)
        tgt_ex = torch.tensor([f.tgt_ex for f in features], dtype=torch.long)
        
        res = {
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'src_token_type_ids': src_token_type_ids,
            'src_dialact_ids': src_dialact_ids,
            'src_emotion_ids': src_emotion_ids,
            'src_len': src_len,
            
            'tgt_input_ids': tgt_input_ids,
            'tgt_token_type_id': tgt_token_type_id,
            'tgt_label_ids': tgt_label_ids,
            'tgt_dialact_id': tgt_dialact_id,
            'tgt_emotion_id': tgt_emotion_id,
            'tgt_len': tgt_len,
            
            'tgt_er': tgt_er,
            'tgt_in': tgt_in,
            'tgt_ex': tgt_ex,
        }
        
        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()
            
            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []
                    
            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch
                
        except StopIteration:
            pass
    
    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog']))), reader))


# for inference
def prepare_infer_batch(features, toker):
    res = FeatureDataset.collate(features, toker)
    
    res['tgt_input_ids'] = res.pop('tgt_input_ids')[:, :1]
    res.pop('tgt_label_ids')
    
    res['batch_size'] = res['src_input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_er': 'pred_er',
        'cls_in': 'pred_in',
        'cls_ex': 'pred_ex',
        'cls_dialact_id': 'pred_dialact_id',
        'cls_emotion_id': 'pred_emotion_id',
    }

    if GOLDEN_CM:
        other_res['cls_er'] = res.get('tgt_er')
        other_res['cls_in'] = res.get('tgt_in')
        other_res['cls_ex'] = res.get('tgt_ex')
    else:
        other_res['cls_er'] = res.pop('tgt_er')
        other_res['cls_in'] = res.pop('tgt_in')
        other_res['cls_ex'] = res.pop('tgt_ex')

    if GOLDEN_DA:
        other_res['cls_dialact_id'] = res.get('tgt_dialact_id')
    else:
        other_res['cls_dialact_id'] = res.pop('tgt_dialact_id')

    if GOLDEN_EM:
        other_res['cls_emotion_id'] = res.get('tgt_emotion_id')
    else:
        other_res['cls_emotion_id'] = res.pop('tgt_emotion_id')
    
    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give max_src_len'
    infer_batch_size = kwargs.get('infer_batch_size')
    
    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    
    features = []
    turn_lens = []
    posts = []
    references = []
    for line in tqdm.tqdm(reader, total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        features.extend(convert_inputs_to_features(inputs, toker, **kwargs))
        turn_lens.append(len(inputs))
        for i in range(len(inputs)):
            ipt = inputs[i]
            posts.append(toker.decode([ee for e in ipt['context'] for ee in e + [toker.eos_token_id]]))
            references.append(toker.decode(ipt['response']))
    
        if len(turn_lens) == infer_batch_size:
            yield prepare_infer_batch(features, toker), posts, references, turn_lens
            features = []
            turn_lens = []
            posts = []
            references = []
            
    if len(turn_lens) > 0:
        yield prepare_infer_batch(features, toker), posts, references, turn_lens
