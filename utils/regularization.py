import torch
import numpy as np
from loguru import logger


class EarlyStopping:
    def __init__(
        self, patience=3, verbose=True, path="check_point.pt", multi_gpus=False
    ):
        """[summary]
        Args:
            patience (int): How many epochs to wait before decrease validation loss
            verbose (bool):
            path (str, optional): Path to save
        """

        self.patience = patience
        self.verbose = verbose
        self._path = path
        self._step = 0
        self._min_val_loss = np.inf
        self._early_stopping = False
        self._multi_gpus = multi_gpus

    def __call__(self, val_loss, model, epoch):
        if self._early_stopping:
            return

        if self._min_val_loss < val_loss:  # val_loss 증가
            if self._step >= self.patience:
                self._early_stopping = True
                if self.verbose:
                    logger.info(
                        f"Validation loss increased for {self.patience} epochs...\t Best_val_loss : {self._min_val_loss}"
                    )
            elif self._step < self.patience:
                self._step += 1
        else:
            self._step = 0
            if self.verbose:
                logger.info(
                    f'Validation loss decreased ({self._min_val_loss:.6f} ---> {val_loss:.6f})\tSaving model..."{self.path}"'
                )
            self._min_val_loss = val_loss
            self._best_epoch = epoch
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        if self._multi_gpus:
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path
