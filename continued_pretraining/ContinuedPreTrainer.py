from typing import Dict, Union, Any

import torch
from packaging import version
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from transformers.trainer import Trainer
from transformers.utils import logging

from continued_pretraining.DGCCA import dgcca_loss
from continued_pretraining.SweDataset import SweDataCollator

# Stuff for fp16
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp

        _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


def embedding_distance_loss(pred, target):
    return mse_loss(pred, target)


class ContinuedPreTrainer(Trainer):
    def __init__(self,
                 dataloader_workers: int = 0,
                 per_device_swe_batch_size: int = 256,
                 swe_data_collator: SweDataCollator = None,
                 lm_dataset: ConcatDataset = None,
                 cca_loss: bool = True,
                 train_cls: bool = False,
                 use_lm: bool = True,
                 use_swe: bool = True,
                 **kwargs):
        """
        :param dataloader_workers:          Number of dataloader workers to use. Applies to both/all dataloaders.
        :param per_device_swe_batch_size:   Batch size for the SWE loss. Must always be present for implementation
                                            reasons. The batch size for MLM is passed to the superclass via
                                            `per_device_batch_size`.
        :param swe_data_collator:           The data collator to be used for the SWE dataset. Must always be present for
                                            implementation reasons. The data collator for MLM is passed to the
                                            superclass as `data_collator`.
        :param lm_dataset:                  The dataset to be used for the LM part of training. Can be `None` if MLM is
                                            not used. This will be an "inner" dataset. The SWE dataset is passed as the
                                            `train_dataset`.
        :param cca_loss:                    When calculating the SWE loss, use DGCCA (default: True). Else, use MSE.
        :param train_cls:                   When calculating the SWE loss, compare to the representation for the CLS
                                            token (default: False). Else, compare to the mean-pooled last-layer token
                                            representations.
        :param use_lm:                      During the training step, calculate MLM loss (default: True).
        :param use_swe:                     During the training step, calculate SWE loss (default: True).
        :param kwargs:                      These remaining arguments are passed on the `Trainer` superclass.
        """
        logger.debug("Initialising trainer")
        super().__init__(**kwargs)

        if not use_swe:
            raise NotImplementedError(
                "You are not using the SWE-based loss this trainer implements. Please use "
                "the standard huggingface Trainer class to do LM-only training.")

        self.cca_loss = cca_loss
        self.train_cls = train_cls
        self.use_swe = use_swe
        self.use_lm = use_lm
        self.swe_data_collator = swe_data_collator
        self.dataloader_workers = dataloader_workers
        self.swe_batch_size = per_device_swe_batch_size

        if use_lm:
            self.lm_dataset = lm_dataset
            self.lm_dataloader = self.get_inner_dataloader(lm_dataset, self.data_collator,
                                                           self.args.per_device_train_batch_size)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        loss = torch.zeros([]).to(self.args.device)
        if self.use_swe:
            swe_loss = self.do_swe_forward(model, inputs)
            self.do_backward(swe_loss)
            swe_loss.detach()
            loss += swe_loss
        if self.use_lm:
            lm_loss = self.do_lm_forward(model)
            self.do_backward(lm_loss)
            lm_loss.detach()
            loss += lm_loss
        return loss.detach()

    def _prepare_lm_inputs(self) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = next(self.step_lm_dataset())

        if inputs is None:
            self.lm_dataloader = self.get_inner_dataloader(self.lm_dataset, self.data_collator,
                                                           self.args.per_device_train_batch_size)
            return self._prepare_lm_inputs()

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        return inputs

    def get_swe_loss(self, last_layer_hidden, mask, target_vectors):
        if self.train_cls:
            pred = last_layer_hidden[:, 0, :]
        else:
            # subword tokens
            pred = _mean_pool(last_layer_hidden, mask)
        if self.cca_loss:
            return dgcca_loss([pred, target_vectors])
        # else
        return embedding_distance_loss(pred, target_vectors)

    def step_lm_dataset(self):
        for i, inputs in enumerate(self.lm_dataloader):
            yield inputs

    def get_inner_dataloader(self,
                             dataset: ConcatDataset,
                             collator: SweDataCollator,
                             batch_size) -> DataLoader:
        train_sampler = RandomSampler(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=self.dataloader_workers,
            collate_fn=collator,
            drop_last=self.args.dataloader_drop_last
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.swe_batch_size,
            sampler=train_sampler,
            num_workers=self.dataloader_workers,
            collate_fn=self.swe_data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def do_lm_forward(self, model):
        inputs = self._prepare_lm_inputs()
        inputs["output_hidden_states"] = False  # reassuring myself here
        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        return loss

    def do_swe_forward(self, model, swe_inputs):
        swe_inputs = self._prepare_inputs(swe_inputs)
        swe_inputs["output_hidden_states"] = True
        swe_targets = swe_inputs.pop("target_vectors")

        if self.args.fp16 and _use_native_amp:
            with autocast():
                swe_outputs = model(**swe_inputs)
        else:
            swe_outputs = model(**swe_inputs)
        swe_loss = self.get_swe_loss(
            swe_outputs["hidden_states"][-1], swe_inputs["attention_mask"],
            swe_targets)

        return swe_loss

    def do_backward(self, loss):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()


def _mean_pool(data, mask):
    return (data * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)
