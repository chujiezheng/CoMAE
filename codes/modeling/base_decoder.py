# coding=utf-8

from typing import Iterable, List

import torch
from torch import Tensor
from modeling.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BaseDecoder(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def get_word_embeddings(self):
        raise NotImplementedError

    def get_input_embeddings(self):
        raise NotImplementedError

    def set_input_embeddings(self, new_embeddings):
        raise NotImplementedError

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        raise NotImplementedError
