# coding=utf-8

import argparse
import json
import shutil
import pickle
import os
import logging
import multiprocessing as mp
from os.path import dirname, exists, join

import torch
import tqdm
from inputter import inputters
from utils.building_utils import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--max_src_turn', type=int, default=None)
parser.add_argument('--max_src_len', type=int, default=150, help='discard data longer than this')
parser.add_argument('--max_tgt_len', type=int, default=None, help='discard data longer than this')
parser.add_argument('--max_knl_len', type=int, default=None, help='discard data longer than this')
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--single_processing', action='store_true', help='do not use multiprocessing')

args = parser.parse_args()

names = {
    'data_name': args.data_name,
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

inputter = inputters[args.inputter_name]()
toker = build_model(only_toker=True, **names)

with open(args.corpus) as f:
    reader = f.readlines()

if not os.path.exists(f'./DATA/{args.data_name}'):
    os.mkdir(f'./DATA/{args.data_name}')
save_dir = f'./DATA/{args.data_name}/{args.inputter_name}.{args.config_name}'
if not exists(save_dir):
    os.mkdir(save_dir)

kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_src_len': args.max_src_len,
    'max_tgt_len': args.max_tgt_len,
    'max_knl_len': args.max_knl_len,
    'label_num': args.label_num,
    'only_encode': args.only_encode,
}

def process_data(line):
    data = json.loads(line)
    inputs = inputter.convert_data_to_inputs(
        data=data,
        toker=toker,
        **kwargs
    )
    features = inputter.convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        **kwargs,
    )
    return features

processed_data = []
if args.single_processing:
    for features in map(process_data, tqdm.tqdm(reader, total=len(reader))):
        processed_data.extend(features)
else:
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for features in pool.imap(process_data, tqdm.tqdm(reader, total=len(reader))):
            processed_data.extend(features)

# save data
data_path = f'{save_dir}/data.pkl'
with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)
kwargs.update({'n_examples': len(processed_data)})
# save relevant information to reproduce
with open(f'{save_dir}/meta.json', 'w') as writer:
    json.dump(kwargs, writer, indent=4)
torch.save(toker, f'{save_dir}/tokenizer.pt')
