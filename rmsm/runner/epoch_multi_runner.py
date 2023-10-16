# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import rmsm
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochMultiRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs[0], dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(1)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(3)  # Prevent possible deadlock during epoch transition
        running_loss1 = 0
        running_loss2 = 0
        running_loss3 = 0
        running_acc1 = 0
        running_acc2 = 0
        running_acc3 = 0
        loader_len = len(data_loader)
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            running_loss1 += self.outputs[0]['loss'].item()
            running_loss2 += self.outputs[1]['loss'].item()
            running_loss3 += self.outputs[2]['loss'].item()
            running_acc1 += self.outputs[0]['log_vars']['top-1']
            running_acc2 += self.outputs[1]['log_vars']['top-1']
            running_acc3 += self.outputs[2]['log_vars']['top-1']
            self.call_hook('after_val_iter')
            del self.data_batch

        self.val_loss1 = running_loss1 / loader_len
        self.val_loss2 = running_loss2 / loader_len
        self.val_loss3 = running_loss3 / loader_len
        self.val_acc1 = running_acc1 / loader_len
        self.val_acc2 = running_acc2 / loader_len
        self.val_acc3 = running_acc3 / loader_len
        # print("打印验证的损失函数")
        # print(self.val_loss1)
        # print(self.val_loss2)
        # print(self.val_loss3)
        # print("打印验证的精度")
        # print(self.val_acc1)
        # print(self.val_acc2)
        # print(self.val_acc3)
        with open(self.work_dir + '/val_loss1.txt', 'a') as f:
            f.write(str(self.val_loss1) + "\n")
            f.close()

        with open(self.work_dir + '/val_loss2.txt', 'a') as f:
            f.write(str(self.val_loss2) + "\n")
            f.close()

        with open(self.work_dir + '/val_loss3.txt', 'a') as f:
            f.write(str(self.val_loss3) + "\n")
            f.close()

        with open(self.work_dir + '/accuracy1_top-1.txt', 'a') as f:
            f.write(str(self.val_acc1) + "\n")
            f.close()

        with open(self.work_dir + '/accuracy1_top-2.txt', 'a') as f:
            f.write(str(self.val_acc2) + "\n")
            f.close()

        with open(self.work_dir + '/accuracy1_top-3.txt', 'a') as f:
            f.write(str(self.val_acc3) + "\n")
            f.close()

        self.val_acc = self.val_acc1 + self.val_acc2 + self.val_acc3
        self.call_hook('after_val_epoch')

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert rmsm.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        self.best_acc = 10.0
        # self.val_loss = 100
        self.val_acc = 0

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

                    if self.val_acc > self.best_acc:
                        if (self.val_acc1 >= 60.0) & (self.val_acc2 >= 60.0) & (self.val_acc3 >= 60.0):
                            time.sleep(1)
                            output_path = self.work_dir
                            save_path = osp.join(output_path, 'checkpoint.pth')
                            latest_pth = osp.join(output_path, 'latest.pth')
                            # 直接复制latest.pth文件
                            shutil.copy(latest_pth, save_path)
                            self.best_acc = self.val_acc
                            print("save best model", self.val_acc1, self.val_acc2, self.val_acc3)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/rmsm/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                rmsm.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
