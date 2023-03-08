from typing import List

import pytorch_lightning as pl
import torch
import numpy as np
import os
import torch.nn as nn
from pytorch_lightning.accelerators import CPUAccelerator, CUDAAccelerator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar


class NSELoss1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, pred):
        denominator = torch.sum(torch.pow(real - torch.mean(real), 2))
        numerator = torch.sum(torch.pow(pred - real, 2))
        return numerator / denominator


class BaseLearner(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 loss_metric=torch.nn.MSELoss(), eval_metric_list=None, lr=0.01, optimizer=None):
        super(BaseLearner, self).__init__()
        if eval_metric_list is None:
            eval_metric_list = [torch.nn.L1Loss()]
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

        self.train_loss_metric = loss_metric
        self.val_loss_metric = loss_metric
        self.train_eval_metric_list = eval_metric_list
        self.val_eval_metric_list = eval_metric_list

    def training_step(self, batch, batch_idx):
        y, y_hat = self.predict_step(batch, batch_idx)
        for metric_idx, log_metric in enumerate(self.train_eval_metric_list):
            self.log('train_metric_' + str(metric_idx + 1), log_metric(y, y_hat),
                     prog_bar=True, on_step=True, on_epoch=False)
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
        x, y, t = batch
        y_hat = self.model.forward(x, t)
        return y, y_hat

    def configure_optimizers(self):
        if self.optimizer is not None:
            return {"optimizer": self.optimizer,
                    "lr_scheduler": torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)}
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
            return {"optimizer": optimizer,
                    "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                    }


def get_trainer(max_epochs, checkpoint_path=None, callbacks=None, logger=True):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=1,
        accelerator=CUDAAccelerator() if torch.cuda.is_available() else CPUAccelerator(),
        enable_checkpointing=True,
        logger=logger,
        default_root_dir=checkpoint_path,
        callbacks=callbacks
    )
    return trainer


def get_callbacks(check_point_path):
    es = EarlyStopping(monitor="train_loss", mode="min", patience=10)
    mc = ModelCheckpoint(dirpath=check_point_path, save_last=False,
                         monitor="train_loss", mode="min", save_top_k=1,
                         filename='{epoch}-{loss:.4f}')
    bar = TQDMProgressBar(refresh_rate=1)
    callbacks = [es, mc, bar]
    return callbacks


def train(learner, train_dataloaders, checkpoint_path, max_epochs=100, val_dataloaders=None):
    file_path = os.path.join(checkpoint_path, 'model_state.pt')
    if os.path.exists(file_path):
        file_path = os.path.join(checkpoint_path, 'model_state.pt')
        model = torch.load(file_path)
        learner.model = model
    else:
        trainer = get_trainer(max_epochs, checkpoint_path, get_callbacks(checkpoint_path))
        trainer.fit(learner, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
        torch.save(learner.model, os.path.join(checkpoint_path, 'model_state.pt'))
        model = learner.model
    return model, learner


def forecast(learner, dataloader):
    with torch.no_grad():
        def handle_predict(result):
            real_list = []
            pred_list = []
            for r in result:
                real, pred = r
                real_list.append(real.cpu().detach().numpy())
                pred_list.append(pred.cpu().detach().numpy())
            return np.concatenate(real_list, axis=0), np.concatenate(pred_list, axis=0)

        trainer = get_trainer(max_epochs=100, callbacks=[], logger=False)
        real_arr, pred_arr = handle_predict(trainer.predict(learner, dataloaders=dataloader))
        return real_arr, pred_arr


def forecast2(model, dataloader, slide_window):
    y_real_list = []
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            y_hat = model.forward(x)
            y_real_list.append(y.to(torch.device('cpu')).detach().numpy())
            y_pred_list.append(y_hat.to(torch.device('cpu')).detach().numpy())
    y_real = y_real_list[0]
    y_pred = y_pred_list[0]

    y_real_re = np.zeros(shape=(len(y_real), len(y_real) + slide_window))
    for i in range(y_real_re.shape[0]):
        y_real_re[i, i:i + slide_window] = y_real[i, :]
    y_real_re[np.where(y_real_re == 0)] = np.nan
    y_real_mean = np.nanmean(y_real_re, axis=0)

    y_pred_re = np.zeros(shape=(len(y_pred), len(y_pred) + slide_window))
    for i in range(y_real_re.shape[0]):
        y_pred_re[i, i:i + slide_window] = y_pred[i, :]
    y_pred_re[np.where(y_pred_re == 0)] = np.nan
    y_pred_mean = np.nanmean(y_pred_re, axis=0)
    return y_real_mean, y_pred_mean, y_real, y_pred


if __name__ == '__main__':
    nse = NSELoss1D()(torch.tensor([1., 2., 3., 4., 5., 6.]), torch.tensor([1., 1.5, 2.5, 3.6, 3.2, 7.8]))
    from sklearn.metrics import r2_score

    print(r2_score([1., 2., 3., 4., 5., 6.], [1., 1.5, 2.5, 3.6, 3.2, 7.8]))
