import os
import pickle
import time
from typing import Optional

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LineByLineTextDatasetCached(Dataset):
    """
    This dataset class is interchangeable in terms of data processing with the `LineByLineTextDataset` from the
    transformers library, with the addition of caching functionality for faster training startup.
    If I was developing the continued pre-training code from scratch now, it would be more appropriate to use the
    huggingface datasets library instead, but as a drop-in replacement, this worked better.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache: bool = False,
                 cache_dir: Optional[str] = None):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cache_dir = cache_dir if cache_dir else directory
        cached_features_file = os.path.join(
            cache_dir,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(self.block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

                batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
                self.examples = batch_encoding["input_ids"]

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        for i, ex in enumerate(self.examples):
            self.examples[i] = np.array(ex)
        self.examples = np.array(self.examples, dtype="object")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
