import glob
import os
import shutil
import unittest

import torch
from torch.utils.data import ConcatDataset
from transformers import TextDataset, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, \
    PreTrainedTokenizer, BertForMaskedLM, AutoConfig, XLMRobertaForMaskedLM

from continued_pretraining.ContinuedPreTrainer import ContinuedPreTrainer
from continued_pretraining.SweDataset import SweDataset, SweDataCollator
from run_continued_pretraining import eval_and_log


class TestIntermediateTrainer(unittest.TestCase):

    def test_defaults(self):
        """ confirm default settings still work as expected """
        model = torch.nn.Identity()
        lm_dataset = self._get_lm_dataset(self.bert_tokenizer)
        trainer = ContinuedPreTrainer(model=model, lm_dataset=lm_dataset, tokenizer=self.bert_tokenizer)
        self.assertTrue(trainer.cca_loss)
        self.assertTrue(trainer.use_lm)
        self.assertTrue(trainer.use_swe)
        self.assertFalse(trainer.train_cls)

    def test_refuses_lm_only(self):
        """ while doing LM-only with the custom trainer might work, it's not the intended use and deliberately
         not supported """
        model = torch.nn.Identity()
        self.assertRaises(NotImplementedError, lambda: ContinuedPreTrainer(model=model, use_swe=False))

    def test_training_steps(self):
        """ check different combinations of data/losses """
        tokenizer = self.bert_tokenizer
        trainer = ContinuedPreTrainer(
            per_device_swe_batch_size=self.batch_size * 4,
            train_dataset=self._get_swe_dataset(tokenizer),
            swe_data_collator=SweDataCollator(tokenizer),
            lm_dataset=self._get_lm_dataset(tokenizer),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
            model=self.mini_bert,
            tokenizer=tokenizer,
            args=self.args
        )
        self._test_training_step(trainer)

        tokenizer = self.xlmr_tokenizer
        trainer = ContinuedPreTrainer(
            per_device_swe_batch_size=self.batch_size * 4,
            train_dataset=self._get_swe_dataset(tokenizer),
            swe_data_collator=SweDataCollator(tokenizer),
            lm_dataset=self._get_lm_dataset(tokenizer),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
            model=self.mini_xlmr,
            tokenizer=tokenizer,
            args=self.args
        )
        self._test_training_step(trainer)

    def _test_training_step(self, trainer):
        train_dataloader = trainer.get_train_dataloader()
        for step, inputs in enumerate(train_dataloader):
            loss = trainer.training_step(trainer.model, inputs)
            break
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(loss.requires_grad)

    def test_inner_dataloader(self):
        """ inner dataloader must wrap around """
        trainer = ContinuedPreTrainer(
            per_device_swe_batch_size=self.batch_size * 4,
            train_dataset=self._get_swe_dataset(self.bert_tokenizer),
            swe_data_collator=SweDataCollator(self.bert_tokenizer),
            lm_dataset=self._get_lm_dataset(self.bert_tokenizer),
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.bert_tokenizer),
            model=torch.nn.Identity(),
            tokenizer=self.bert_tokenizer,
            args=self.args
        )
        for i in range(len(trainer.lm_dataloader) * 4):
            inputs = trainer._prepare_lm_inputs()
            self.assertIsInstance(inputs["input_ids"], torch.Tensor)
            self.assertIsInstance(inputs["labels"], torch.Tensor)

    def test_eval_and_log(self):
        """ run eval successfully and output a file with perplexity """

        trainer = self._get_basic_trainer(self.bert_tokenizer, self.mini_bert)
        results = {}
        eval_and_log(results, trainer, self.args)
        self.assertTrue(os.path.exists("tmp_trainer/eval_results_lm.txt"))
        self.assertIsNotNone(results["perplexity"])

    def test_save_model(self):
        """ confirm model is saved with tokeniser, optimiser and state dict """

        trainer = self._get_basic_trainer(self.bert_tokenizer, self.mini_bert)
        trainer.save_model(os.path.join(self.args.output_dir, "bert"))
        self._expect_output_files(
            ["bert/config.json", "bert/vocab.txt", "bert/tokenizer.json", "bert/tokenizer_config.json",
             "bert/pytorch_model.bin", "bert/training_args.bin", "bert/special_tokens_map.json"])

        trainer = self._get_basic_trainer(self.xlmr_tokenizer, self.mini_xlmr)
        trainer.save_model(os.path.join(self.args.output_dir, "xlmr"))
        self._expect_output_files(["xlmr/config.json", "xlmr/pytorch_model.bin", "xlmr/sentencepiece.bpe.model",
                                   "xlmr/special_tokens_map.json", "xlmr/tokenizer.json", "xlmr/tokenizer_config.json",
                                   "xlmr/training_args.bin"])

    def _get_basic_trainer(self, tokenizer, model):
        return ContinuedPreTrainer(
            per_device_swe_batch_size=self.batch_size * 4,
            train_dataset=self._get_swe_dataset(tokenizer),
            swe_data_collator=SweDataCollator(tokenizer),
            use_lm=False,
            eval_dataset=self._get_lm_dataset(tokenizer),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
            model=model,
            tokenizer=tokenizer,
            args=self.args
        )

    def _expect_output_files(self, files):
        for file in files:
            self.assertTrue(os.path.exists(os.path.join(self.args.output_dir, file)))

    def _get_lm_dataset(self, tokenizer: PreTrainedTokenizer):
        return ConcatDataset([TextDataset(file_path=self.lm_file_name, tokenizer=tokenizer, block_size=128)])

    def _get_swe_dataset(self, tokenizer: PreTrainedTokenizer):
        return ConcatDataset([SweDataset(file_path=self.swe_file_name, tokenizer=tokenizer, overwrite_cache=False)])

    @classmethod
    def setUpClass(cls) -> None:
        cls.lm_file_name = "he-mini.txt"
        cls.swe_file_name = "en_fr_mapped_small.txt"
        cls.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        cls.bert_config = AutoConfig.from_pretrained("bert-base-multilingual-cased")
        cls.bert_config.num_hidden_layers = 1
        cls.mini_bert = BertForMaskedLM(config=cls.bert_config)
        cls.xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        cls.xlmr_config = AutoConfig.from_pretrained("xlm-roberta-base")
        cls.xlmr_config.num_hidden_layers = 1
        cls.mini_xlmr = XLMRobertaForMaskedLM(cls.xlmr_config)
        cls.batch_size = 1
        cls.args = TrainingArguments(output_dir="tmp_trainer",
                                     per_device_train_batch_size=cls.batch_size,
                                     per_device_eval_batch_size=cls.batch_size,
                                     no_cuda=True,
                                     report_to=["none"])

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("tmp_trainer")
        for f in glob.glob("cached_*_*"):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
