import copy
import html
import os
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SweDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, overwrite_cache: bool, cache_dir: str = None):
        logger.info("Reading phrase2vec data from file at %s", file_path)

        directory, filename = os.path.split(file_path)
        embeddings_name = os.path.normpath(directory).split(os.sep)[-1]
        cached_filename = "cached_{}_{}".format(embeddings_name, filename)
        cache_dir = cache_dir if cache_dir else directory
        cached_path = os.path.join(cache_dir, cached_filename)
        assert os.path.isfile(file_path) or os.path.isfile(cached_path), f"Input file path {file_path} not found"

        if (not os.path.exists(cached_path)) or overwrite_cache:
            vectors = KeyedVectors.load_word2vec_format(file_path, binary=False, limit=200000)
            vectors.init_sims(replace=True)  # this is deprecated in gensim but works for now
            vectors.save(cached_path)

        self.examples = KeyedVectors.load(cached_path, mmap='r')
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector = self.examples.vectors[index]
        target = copy.deepcopy(vector)
        entity = self.examples.index_to_key[index]
        input_ids, attention_mask = self.encode_example(entity)
        return input_ids, attention_mask, target

    def __len__(self) -> int:
        return len(self.examples.vectors)

    def encode_example(self, entity) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer(html.unescape(entity), max_length=20, truncation=True)
        input_ids = np.array(encoded.input_ids)
        attention_mask = np.array(encoded.attention_mask)
        return input_ids, attention_mask


@dataclass
class SweDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
                 ) -> Dict[str, torch.Tensor]:
        input_ids, attention_masks, target_vectors = map(list, zip(*examples))

        return {
            "input_ids": self._tensorize_batch(input_ids),
            "attention_mask": self._tensorize_batch(attention_masks),
            "target_vectors": self._tensorize_batch(target_vectors)
        }

    def _tensorize_batch(self, examples: List[Union[torch.Tensor, np.ndarray]]) -> torch.Tensor:
        examples = [torch.tensor(ex) if not isinstance(ex, torch.Tensor) else ex for ex in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
