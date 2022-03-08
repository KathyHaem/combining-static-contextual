import glob
import os
import unittest

from transformers import AutoConfig, BertForMaskedLM, XLMRobertaForMaskedLM, BertPreTrainedModel, RobertaPreTrainedModel

from run_continued_pretraining import freeze_layers


class TestScript(unittest.TestCase):
    def test_freeze_layers(self):
        self.assertTrue(isinstance(self.mini_bert, BertPreTrainedModel))
        freeze_layers(self.mini_bert, 1)
        for param in self.mini_bert.bert.encoder.layer[0].parameters():
            self.assertFalse(param.requires_grad)
        self.mini_bert.requires_grad_(True)
        self.assertTrue(isinstance(self.mini_xlmr, RobertaPreTrainedModel))
        freeze_layers(self.mini_xlmr, 1)
        for param in self.mini_xlmr.roberta.encoder.layer[0].parameters():
            self.assertFalse(param.requires_grad)
        self.mini_xlmr.requires_grad_()

    @classmethod
    def setUpClass(cls) -> None:
        cls.bert_config = AutoConfig.from_pretrained("bert-base-multilingual-cased")
        cls.bert_config.num_hidden_layers = 2
        cls.mini_bert = BertForMaskedLM(config=cls.bert_config)

        cls.xlmr_config = AutoConfig.from_pretrained("xlm-roberta-base")
        cls.xlmr_config.num_hidden_layers = 2
        cls.mini_xlmr = XLMRobertaForMaskedLM(cls.xlmr_config)

    @classmethod
    def tearDownClass(cls) -> None:
        for f in glob.glob("cached_*_*"):
            os.remove(f)
        if os.path.exists("cached_swe_en_fr_mapped_small.txt"):
            os.remove("cached_swe_en_fr_mapped_small.txt")


if __name__ == '__main__':
    unittest.main()
