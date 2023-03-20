import torch
import torch.nn as nn


class NSELoss(nn.Module):
    def __init__(self):
        """
        这个是最原版的NSE，但对于分batch训练，nse的计算结果常常出现较大差异
        """
        super().__init__()

    def forward(self, real, pred):
        denominator = torch.sum(torch.pow(real - torch.mean(real), 2))
        numerator = torch.sum(torch.pow(pred - real, 2))
        return numerator / denominator


class RAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, pred):
        denominator = torch.sum(torch.abs(real - torch.mean(real)))
        numerator = torch.sum(torch.abs(pred - real))
        return numerator / denominator


class AdaptiveNSE(nn.Module):
    """
    将M0预测误差为分母(原NSE是将平均计算器为误差)
    """
    def __init__(self):
        super().__init__()

    def forward(self, real, pred, proto):
        denominator = torch.sum(torch.pow(real - proto, 2))
        numerator = torch.sum(torch.pow(real - pred, 2))
        return numerator / denominator


class NSELossFixedMeanWarmUp(nn.Module):
    def __init__(self, real_mean, warmup_length):
        """
        固定均值的NSE,加入warm up length
        """
        super().__init__()
        self.real_mean = torch.tensor(real_mean)
        self.warmup_length = warmup_length

    def forward(self, real, pred):
        denominator = torch.sum(torch.pow(real[self.warmup_length:] - self.real_mean, 2))
        numerator = torch.sum(torch.pow(pred[self.warmup_length:] - real[self.warmup_length:], 2))
        return numerator / denominator


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, pred):
        return torch.sum(torch.log(torch.cosh(pred - real)))


class QuantileLoss(nn.Module):

    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, real, pred):
        pred_underflow = real - pred
        q_loss = torch.max(self.q * pred_underflow, (self.q - 1) * pred_underflow)
        return torch.mean(q_loss)
