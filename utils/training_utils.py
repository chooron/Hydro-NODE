from typing import List

import pytorch_lightning as pl
import torch
import numpy as np
import os
import torch.nn as nn
from pytorch_lightning.accelerators import CPUAccelerator, CUDAAccelerator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from torchmetrics import MeanSquaredError, Metric, MeanAbsoluteError


class BaseLearner(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 loss_metric: Metric = MeanSquaredError(), eval_metric_list=None):
        super(BaseLearner, self).__init__()
        if eval_metric_list is None:
            eval_metric_list = [MeanAbsoluteError()]
        self.model = model

        self.train_loss_metric = loss_metric
        self.val_loss_metric = loss_metric
        self.train_eval_metric_list = eval_metric_list
        self.val_eval_metric_list = eval_metric_list

    def training_step(self, batch, batch_idx):
        y, y_hat = self.predict_step(batch, batch_idx)
        for metric_idx, log_metric in enumerate(self.train_eval_metric_list):
            self.log('train_metric_' + str(metric_idx + 1), log_metric(y, y_hat), prog_bar=True, on_step=True,
                     on_epoch=False)
        train_loss = self.train_loss_metric(y, y_hat)
        self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=False)
        return train_loss

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.predict_step(batch, batch_idx)
        for metric_idx, log_metric in enumerate(self.val_eval_metric_list):
            self.log('val_metric_' + str(metric_idx + 1), log_metric(y, y_hat),
                     prog_bar=True, on_step=True, on_epoch=True)
        val_loss = self.val_loss_metric(y, y_hat)
        self.log('val_loss', val_loss, prog_bar=True, on_step=True, on_epoch=True)
        return val_loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device).squeeze(1)
        y_hat = self.model.forward(x)
        return y, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-3)
        return {"optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)}


def get_trainer(max_epochs, checkpoint_path=None, callbacks=None):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=1,
        accelerator=CUDAAccelerator() if torch.cuda.is_available() else CPUAccelerator(),
        enable_checkpointing=True,
        logger=False,
        default_root_dir=checkpoint_path,
        callbacks=callbacks
    )
    return trainer


def get_callbacks(check_point_path):
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    mc = ModelCheckpoint(dirpath=check_point_path, save_last=False,
                         monitor="val_loss", mode="min", save_top_k=1,
                         filename='{epoch}-{val_loss:.4f}')
    bar = TQDMProgressBar(refresh_rate=1)
    callbacks = [es, mc, bar]
    return callbacks


def train(model, learner, train_val_datamodule, checkpoint_path, train_idx=0, max_epochs=100):
    temp_checkpoint_path = os.path.join(checkpoint_path, 'train_history', 'train_{}'.format(train_idx))
    file_path = os.path.join(temp_checkpoint_path, 'model_state.pt')
    if os.path.exists(file_path):
        file_path = os.path.join(temp_checkpoint_path, 'model_state.pt')
        model.load_state_dict(torch.load(file_path))
        learner.model = model
    else:
        train_dataloader, val_dataloader = train_val_datamodule.get_sample()
        trainer = get_trainer(max_epochs, temp_checkpoint_path, get_callbacks(temp_checkpoint_path))
        trainer.fit(learner, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        torch.save(learner.model.state_dict(), os.path.join(temp_checkpoint_path, 'model_state.pt'))
        model = learner.model
    return model, learner


def forecast(learner, datamodule):
    def handle_predict(result):
        real_list = []
        pred_list = []
        for r in result:
            pred, real = r
            real_list.append(real.cpu().detach().numpy())
            pred_list.append(pred.cpu().detach().numpy())
        return np.concatenate(real_list, axis=0), np.concatenate(pred_list, axis=0)

    trainer = get_trainer(max_epochs=100, callbacks=[])
    datamodule.datamodule_type = 'test'
    dataloader = datamodule.get_sample()
    real_arr, pred_arr = handle_predict(trainer.predict(learner, dataloaders=dataloader))
    if datamodule.target_scaler is not None:
        real_arr = datamodule.target_scaler.inverse_transform(real_arr)
        pred_arr = datamodule.target_scaler.inverse_transform(pred_arr)
    return real_arr, pred_arr
