import os
import timeit
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from continued_pretraining.SweDataset import SweDataset, SweDataCollator


class TestSweDataset(unittest.TestCase):
    def test_load(self):
        self._test_load(self.bert_tokenizer)
        self._test_load(self.xlmr_tokenizer)

    def _test_load(self, tokenizer: PreTrainedTokenizer):
        dataset = SweDataset(tokenizer, file_path=self.file_name, overwrite_cache=True)
        self.assertIsNotNone(dataset.examples)
        self.assertIsNotNone(dataset.tokenizer)
        self.assertEqual(len(dataset), 124)

    def test_reload_from_cache(self):
        """ confirm that loading from cache runs significantly faster """
        timing_new = timeit.timeit(
            lambda: SweDataset(self.bert_tokenizer, file_path=self.file_name, overwrite_cache=True), number=5)
        timing_cached = timeit.timeit(
            lambda: SweDataset(self.bert_tokenizer, file_path=self.file_name, overwrite_cache=False), number=5)
        self.assertGreater(timing_new / 10, timing_cached)

    def test_encode_long_item(self):
        item = "This is a long example that will tokenize into more than twenty (20) tokens using either tokenizer."
        self._test_encode_long_item(self.bert_tokenizer, item)
        self._test_encode_long_item(self.xlmr_tokenizer, item)

    def _test_encode_long_item(self, tokenizer: PreTrainedTokenizer, item: str):
        encoded = tokenizer(item)
        self.assertGreater(len(encoded.input_ids), 20)
        dataset = SweDataset(tokenizer, self.file_name, False)
        inputs, mask = dataset.encode_example(item)
        self.assertEqual(len(inputs), 20)
        self.assertEqual(len(mask), 20)

    def test_get_item(self):
        """ get an item; confirm length of input ids and attention mask """
        self._test_get_item(self.bert_tokenizer)
        self._test_get_item(self.xlmr_tokenizer)

    def _test_get_item(self, tokenizer: PreTrainedTokenizer):
        dataset = SweDataset(tokenizer, self.file_name, False)
        input_ids, attention_mask, target = dataset[96]
        self.assertEqual(len(input_ids), len(attention_mask))
        self.assertIsInstance(input_ids, np.ndarray)
        self.assertIsInstance(attention_mask, np.ndarray)
        self.assertIsInstance(target, np.ndarray)
        tokenizer.decode(input_ids)

    def test_collation(self):
        """ get dataloader and data collator; confirm loading, attributes, shape """
        self._test_collation(self.bert_tokenizer, batch_size=4)
        self._test_collation(self.xlmr_tokenizer, batch_size=4)

    def _test_collation(self, tokenizer, batch_size):
        dataset = SweDataset(tokenizer, self.file_name, False)
        collator = SweDataCollator(tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collator,
            drop_last=True,
        )
        self.assertEqual(len(dataloader), len(dataset) // batch_size)
        batch = next(iter(dataloader))
        self.assertIsInstance(batch["input_ids"], torch.Tensor)
        self.assertIsInstance(batch["attention_mask"], torch.Tensor)
        self.assertIsInstance(batch["target_vectors"], torch.Tensor)
        self.assertEqual(list(batch["input_ids"].size())[0], 4)
        self.assertEqual(list(batch["attention_mask"].size())[0], 4)
        self.assertEqual(list(batch["target_vectors"].size()), [4, 768])

    @classmethod
    def setUpClass(cls) -> None:
        cls.file_name = "en_fr_mapped_small.txt"
        cls.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        cls.xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    @classmethod
    def tearDownClass(cls) -> None:
        """ remove cache """
        os.remove("cached_._en_fr_mapped_small.txt")


if __name__ == '__main__':
    unittest.main()
