# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

"""Adapted from the run_glue.py and run_language_modeling.py fine-tuning scripts by Katharina Hämmerl."""

import logging
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Optional, Tuple, Union

import math

import numpy as np
from torch.utils.data import ConcatDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from transformers.trainer import Trainer
from continued_pretraining.ContinuedPreTrainer import ContinuedPreTrainer
from continued_pretraining.LineByLineTextDatasetCached import LineByLineTextDatasetCached
from continued_pretraining.SweDataset import SweDataset, SweDataCollator

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    update_layers: Optional[int] = field(
        default=-1, metadata={"help": "Fine-tune only the last n layers. "
                                      "Default: -1 (update the whole model)"}
    )
    freeze_embeddings: Optional[bool] = field(
        default=False, metadata={"help": "Whether to keep embeddings frozen during training. "
                                         "Default: False (update embeddings with the rest of the model)."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        }
    )

    train_swe_files: str = field(
        default=None, metadata={"help": "Path to the multilingual embeddings to use (multiple files in glob format."}
    )

    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated to blocks of this size for training."
                    "Defaults to the model max input length for single sentence inputs "
                    "(take into account special tokens)."
        },
    )

    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store dataset caches."}
    )

    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    dataloader_workers: Optional[int] = field(
        default=0, metadata={"help": "How many workers to use in the PyTorch dataloaders. Defaults to 0, which means "
                                     "data loading happens on main thread."}
    )

    use_swe: Optional[bool] = field(
        default=True, metadata={"help": "If False, leave out the SWE part of training."}
    )

    use_lm: Optional[bool] = field(
        default=True, metadata={"help": "If False, leave out the MLM part of training."}
    )

    line_by_line: Optional[bool] = field(
        default=False, metadata={"help": "Whether to read the training data files as LineByLineTextDataset."
                                         "Default: False (read them as TextDataset)."
                                         "Dataset features are cached in both cases."}
    )

    per_device_swe_batch_size: Optional[int] = field(
        default=256, metadata={"help": "Train batch size for the SWE part of training. "
                                       "Can be much larger than for LM because of the smaller effective block size."}
    )

    use_cca_loss: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use CCA loss during the SWE part of training."
                                        "If False, will use MSE loss."}
    )

    train_cls: Optional[bool] = field(
        default=False, metadata={"help": "Whether to calculate the SWE loss using the CLS representation."
                                         "If False, will compare to average over last-layer representations of tokens."}
    )


def get_dataset(args: DataTrainingArguments,
                tokenizer: PreTrainedTokenizer,
                evaluate: bool = False
                ) -> Union[TextDataset, LineByLineTextDatasetCached, ConcatDataset]:
    def _dataset(file_path):
        logger.info("Reading dataset from file: {}".format(file_path))
        if args.line_by_line:
            return LineByLineTextDatasetCached(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=args.data_cache_dir
            )
        else:
            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=args.data_cache_dir)
            for i, ex in enumerate(dataset.examples):
                dataset.examples[i] = np.array(ex)
            dataset.examples = np.array(dataset.examples)
            return dataset

    if evaluate:
        return _dataset(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file)


def get_swe_dataset(tokenizer: PreTrainedTokenizer,
                    data_args: DataTrainingArguments,
                    training_args: TrainingArguments
                    ) -> Union[ConcatDataset, None]:
    def _swe_dataset(file):
        return SweDataset(tokenizer=tokenizer, file_path=file, overwrite_cache=data_args.overwrite_cache,
                          cache_dir=data_args.data_cache_dir)

    if not training_args.do_train or not data_args.use_swe:
        return None
    return ConcatDataset([_swe_dataset(f) for f in glob(data_args.train_swe_files)])


def eval_and_log(results: dict, trainer: Trainer, training_args: TrainingArguments):
    """ Evaluation after end of training, writes perplexity result to file. """
    logger.info("*** Evaluate ***")
    eval_output = trainer.evaluate()
    perplexity = math.exp(eval_output["eval_loss"])
    result = {"perplexity": perplexity}
    output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    results.update(result)


def train_and_save(model_args: ModelArguments, trainer: Trainer):
    resume_from_checkpoint = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    # save again at end of training (even if there were checkpoints saved during training)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()


def get_lm_datasets(data_args: DataTrainingArguments,
                    tokenizer: PreTrainedTokenizer,
                    training_args: TrainingArguments
                    ) -> Tuple[DataCollatorForLanguageModeling,
                               Union[LineByLineTextDatasetCached, TextDataset],
                               ConcatDataset]:
    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if (training_args.do_train and data_args.use_lm) else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability
    )
    return data_collator, eval_dataset, train_dataset


def load_model_and_tokenizer(data_args: DataTrainingArguments,
                             model_args: ModelArguments
                             ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """ Load pretrained model and tokenizer.

    Distributed training:
    The .from_pretrained methods guarantee that only one local process can concurrently
    download model & vocab.
    """
    if not model_args.model_name_or_path:
        raise ValueError("Please specify the base model. This script supports BERT-like models.")

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=model_args.cache_dir,
                                              use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    model.resize_token_embeddings(len(tokenizer))
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    if model_args.update_layers > -1:
        freeze_layers(model, model_args.update_layers)
    if model_args.freeze_embeddings:
        model.base_model.embeddings.requires_grad_(False)

    return model, tokenizer


def freeze_layers(model, update_layers: int):
    if update_layers == 0:
        model.base_model.encoder.requires_grad_(False)
    else:
        for layer in model.base_model.encoder.layer[:-update_layers]:
            layer.requires_grad_(False)


def parse_args() -> Tuple[DataTrainingArguments, ModelArguments, TrainingArguments]:
    """ See all possible arguments in src/transformers/training_args.py
    or by passing the --help flag to this script.
    We now keep distinct sets of args, for a cleaner separation of concerns.
    """
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    # Set seed
    set_seed(training_args.seed)
    return data_args, model_args, training_args


def setup_logging(training_args: TrainingArguments) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)


def main():
    data_args, model_args, training_args = parse_args()
    training_args.prediction_loss_only = True
    setup_logging(training_args)
    model, tokenizer = load_model_and_tokenizer(data_args, model_args)
    data_collator, eval_dataset, train_dataset = get_lm_datasets(data_args, tokenizer, training_args)

    swe_dataset, swe_data_collator = None, None
    if data_args.use_swe:
        swe_dataset = get_swe_dataset(tokenizer, data_args, training_args)
        swe_data_collator = SweDataCollator(tokenizer)

    if data_args.use_lm and not data_args.use_swe:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
    else:
        # Initialize our Trainer
        trainer = ContinuedPreTrainer(
            lm_dataset=train_dataset,
            swe_data_collator=swe_data_collator,
            dataloader_workers=data_args.dataloader_workers,
            per_device_swe_batch_size=data_args.per_device_swe_batch_size,
            cca_loss=data_args.use_cca_loss,
            train_cls=data_args.train_cls,
            use_lm=data_args.use_lm,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=swe_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

    if training_args.do_train:
        train_and_save(model_args, trainer)

    results = {}
    if training_args.do_eval:
        eval_and_log(results, trainer, training_args)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
