# coding=utf-8

import json
import os
import logging
import torch
from os.path import join

from torch.distributed import get_rank

from tokenization import tokers
from modeling import encoders, decoders
from framework import frameworks

logger = logging.getLogger(__name__)


def build_model(only_toker=False, checkpoint=None, local_rank=-1, **kwargs):
    assert 'data_name' in kwargs
    assert 'config_name' in kwargs
    data_name = kwargs.pop('data_name')
    config_name = kwargs.pop('config_name')
    
    if not os.path.exists(f'./CONFIG/{data_name}/{config_name}.json'):
        raise ValueError
    
    with open(f'./CONFIG/{data_name}/{config_name}.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    assert 'tokenizer_type' in config and 'tokenizer_path' in config, 'you should assign a tokenizer with its path'
    assert 'model_type' in config, 'you should assign a model'
    if 'encoder_type' not in config and 'decoder_type' not in config:
        logger.warning(
            'you do not assign either encoder or decoder, please check your framework'
        )
    
    token_id_dict = {
        'bos_token_id': None,
        'eos_token_id': None,
        'pad_token_id': None,
        'unk_token_id': None,
        'cls_token_id': None,
        'sep_token_id': None,
    }
    
    decoder_config_dict = None
    if 'decoder_config' in config:
        decoder_config_dict = config.pop('decoder_config')
    
    if decoder_config_dict is not None:
        for key in token_id_dict.keys():
            if key in decoder_config_dict:
                if token_id_dict[key] is not None:
                    assert token_id_dict[key] == decoder_config_dict[key]
                else:
                    token_id_dict[key] = decoder_config_dict[key]
    
    
    encoder_config_dict = None
    if 'encoder_config' in config:
        encoder_config_dict = config.pop('encoder_config')
    elif config.pop('share_config', False):
        encoder_config_dict = decoder_config_dict.copy()
    
    if decoder_config_dict is None and encoder_config_dict is not None:
        for key in token_id_dict.keys():
            if key in encoder_config_dict:
                if token_id_dict[key] is not None:
                    assert token_id_dict[key] == encoder_config_dict[key]
                else:
                    token_id_dict[key] = encoder_config_dict[key]
    
    if decoder_config_dict is not None and encoder_config_dict is not None:
        for key in token_id_dict.keys():
            if key in encoder_config_dict:
                assert token_id_dict[key] == encoder_config_dict[key]
    
    
    if not only_toker and (local_rank == -1 or get_rank() == 0):
        logger.info('Overall Config Argument Information')
        for a in config:
            logger.info('%-28s  %s' % (a, config[a]))
        if encoder_config_dict is not None:
            logger.info('Encoder Config Argument Information')
            for a in encoder_config_dict:
                logger.info('%-28s  %s' % (a, encoder_config_dict[a]))
        if decoder_config_dict is not None:
            logger.info('Decoder Config Argument Information')
            for a in decoder_config_dict:
                logger.info('%-28s  %s' % (a, decoder_config_dict[a]))
    
    toker = build_tokenizer(
        type=config.get('tokenizer_type'),
        config_path=config.get('tokenizer_path'),
    )
    
    if 'tokenizer_expand_vocab' in config:
        assert isinstance(config['tokenizer_expand_vocab'], list)
        toker.add_tokens(config['tokenizer_expand_vocab'], special_tokens=True)

    if local_rank == -1 or get_rank() == 0:
        for key in token_id_dict.keys():
            if hasattr(toker, key):
                if getattr(toker, key) != token_id_dict[key]:
                    setattr(toker, '_' + key[:-3], token_id_dict[key])
                    logger.info(f'{key} in toker is inconsistent with model')
    
    if only_toker:
        return toker
    
    if config['model_type'].lower() not in frameworks.keys():
        raise ValueError(f"now the model type {config['model_type']} is not supported")
    else:
        Model = frameworks[config['model_type']]
    #model_type = config.pop('model_type')
    
    encoder_checkpoint = config.pop('encoder_checkpoint', None)
    decoder_checkpoint = config.pop('decoder_checkpoint', None)
    if checkpoint is not None and checkpoint.lower() != "none":
        encoder_checkpoint = None
        decoder_checkpoint = None
    
    encoder_config, encoder = build_encoder(
        type=config.pop('encoder_type', None),
        config_dict=encoder_config_dict,
        checkpoint=encoder_checkpoint,
        local_rank=local_rank,
    )
    decoder_config, decoder = build_decoder(
        type=config.pop('decoder_type', None),
        config_dict=decoder_config_dict,
        checkpoint=decoder_checkpoint,
        local_rank=local_rank,
    )
    
    if local_rank == -1 or get_rank() == 0:
        logger.info('building model...')
    model = Model(encoder=encoder, decoder=decoder, toker=toker, **config)
    load_model(model, checkpoint, local_rank=local_rank,)
    return toker, model, encoder_config, decoder_config
    

def build_tokenizer(type, config_path):
    if type.lower() not in tokers.keys():
        raise ValueError(f"now the tokenizer type {type} is not supported")
    else:
        toker = tokers[type.lower()].from_pretrained(config_path)
    
    return toker


def build_encoder(type=None, config_dict=None, checkpoint=None, local_rank=-1):
    if type is None or type.lower() == 'none':
        return None, None
    
    if type.lower() not in encoders.keys():
        raise ValueError(f"now the encoder type {type} is not supported")
    else:
        Config, Model = encoders[type.lower()]
    
    if local_rank == -1 or get_rank() == 0:
        logger.info('building encoder...')
    config = Config.from_dict(config_dict)
    model = Model(config)
    load_model(model, checkpoint, local_rank=local_rank)
    return config, model


def build_decoder(type=None, config_dict=None, checkpoint=None, local_rank=-1):
    if type is None or type.lower() == 'none':
        return None, None
    
    if type.lower() not in decoders.keys():
        raise ValueError(f"now the decoder type {type} is not supported")
    else:
        Config, Model = decoders[type.lower()]

    if local_rank == -1 or get_rank() == 0:
        logger.info('building decoder...')
    config = Config.from_dict(config_dict)
    model = Model(config)
    load_model(model, checkpoint, local_rank=local_rank)
    return config, model


def load_model(model, checkpoint, local_rank=-1):
    if checkpoint is not None and checkpoint.lower() != "none":
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        model_state_dict = torch.load(checkpoint)
        
        model_state_dict = fix_state_dict_namespace(model_state_dict, local_rank)
        if local_rank == -1 or get_rank() == 0:
            logger.info('loading finetuned model from %s' % checkpoint)
        
        strict = False
        if hasattr(model, 'transformer') and all(not e.startswith('transformer.') for e in model_state_dict.keys()):
            model = model.transformer
        if hasattr(model, 'tower') and model.tower:
            strict = True
        
        needed_keys = set(dict(model.named_parameters()).keys())
        loaded_keys = []
        for k, v in model_state_dict.items():
            if k not in needed_keys:
                continue
            try:
                model.load_state_dict({k: v}, strict=False)
                #if local_rank == -1 or get_rank() == 0:
                #    logger.info(' parameter [%s] loaded!' % k)
                loaded_keys.append(k)
            except RuntimeError as e:
                if local_rank == -1 or get_rank() == 0:
                    logger.info(' ??? unmatched parameter [%s]' % k)
                if strict:
                    raise e
        
        loaded_keys = set(loaded_keys)
        missed_keys = needed_keys - loaded_keys

        if local_rank == -1 or get_rank() == 0:
            if len(missed_keys) > 0:
                for k in sorted(missed_keys):
                    logger.info(' !!! parameter [%s] missed' % k)


def deploy_model(model, args, local_rank=-1):
    if local_rank == -1 or get_rank() == 0:
        logger.info('deploying model...')
    n_gpu = args.n_gpu
    device = args.device
    model.to(device)
    
    #if args.local_rank != -1:
    #    model = torch.nn.parallel.DistributedDataParallel(
    #        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #    ).to(args.device)
    #el
    if n_gpu > 1:
        if local_rank == -1 or get_rank() == 0:
            logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    
    return model


def fix_state_dict_namespace(model_state_dict, local_rank=-1):
    old_keys = []
    new_keys = []
    for t in list(model_state_dict.keys()).copy():
        new_key = t
        
        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        elif new_key.startswith('model.'):
            new_key = new_key.replace('model.', '')
        
        if new_key.endswith('.beta'):
            new_key = new_key.replace('.beta', '.bias')
        elif new_key.endswith('.gamma'):
            new_key = new_key.replace('.gamma', '.weight')
        
        old_keys.append(t)
        new_keys.append(new_key)
        
        #for requirement, mapping in zip(added_keys_requirements, added_keys_mappings):
        #    if all(r in new_key for r in requirement):
        #        for ori, new in mapping:
        #            #logger.info(new_key + '->' + new_key.replace(ori, new))
        #            added_key = new_key.replace(ori, new)
        #            model_state_dict[added_key] = model_state_dict[t].clone()

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)
    
    if 'shared.weight' in new_keys and 'encoder.embed_tokens.weight' not in new_keys:
        model_state_dict['encoder.embed_tokens.weight'] = model_state_dict['shared.weight'].clone()
        if local_rank == -1 or get_rank() == 0:
            logger.info(' cloning [encoder.embed_tokens.weight] from [shared.weight]...')
    if 'shared.weight' in new_keys and 'decoder.embed_tokens.weight' not in new_keys:
        model_state_dict['decoder.embed_tokens.weight'] = model_state_dict['shared.weight'].clone()
        if local_rank == -1 or get_rank() == 0:
            logger.info(' cloning [decoder.embed_tokens.weight] from [shared.weight]...')

    return model_state_dict


