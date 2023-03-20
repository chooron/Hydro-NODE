# Hydrology ODE neural network and Loss Function

## Julia环境的预测精度
| 测站编号    | 模型   | 损失函数 |                            精度                            |
|---------|------|:----:|:--------------------------------------------------------:|
| 1013500 | M100 | RSE  | NSE: train=0.948, test=0.897, MSE: train=0.19, test=0.43 |


## 当前的问题

### Hydro-NODE问题

1. 完整序列输入时，参数更新慢，模型参数寻优过程相当缓慢，因此需要提前训练好子模型带入模型中
2. solver中atol和rtol参数对于模型预测精度有着显著影响(1e-3和1e-6)
3. 原本代码中存在过拟合问题

### Batch Training问题

1. Batch training首先需要确定ODE方程的初始值问题，当前仍以M0求解结果为准，但实际上在同一损失函数下，初值问题影响不大
2. Batch training无法使用RSE损失函数，需要额外选取一个损失函数得到与之相似结果，当前使用MAE结果最优，但与RSE仍相差0.3
3. Batch size的选取是一个比较大的问题，如何选取一个合适的batch size还没有一个明确的定论

**Hydro-NODE更多的是一种模型修正，而并非是独立的模型**

## Hydro-NODE与损失函数实验

### 实验内容

1. 损失函数部分选取：

实验设置，测站：1013500，模型：M50， 求解器：torchdiffeq，输入为全序列输入，优化器：AdamW(lr=0.01), with schedular
想法：RSE方法中将y_mean替换为M0的实测流量，待选方法还有quantile loss


### 实验记录

| 测站编号    |   损失函数   |                            精度                            |
|---------|:--------:|:--------------------------------------------------------:|
| 1013500 |   MAE    | NSE: train=0.865, test=0.830, MSE: train=0.60, test=0.71 |
| 1013500 | Huber(1) | NSE: train=0.85, test=0.83, MSE: train=0.556, test=0.71  |
| 1013500 | Huber(2) |  NSE: train=0.80, test=0.83, MSE: train=0.73, test=0.74  |
| 1013500 |   MSE    |  NSE: train=0.85, test=0.84, MSE: train=0.50, test=0.49  |
| 1013500 |   RSE    |  NSE: train=0.90, test=0.86, MSE: train=0.35, test=0.58  |
| 6431500 |   RSE    | NSE: train=-0.03, test=-0.68, MSE: train=0.10, test=0.06 |

2. 批训练样本大小选取：

| 测站编号    | 模型   |     损失函数     | 样本大小 |                             精度                             |
|---------|------|:------------:|------|:----------------------------------------------------------:|
| 1013500 | M50  |     MAE      | 365  | NSE: train=0.895, test=0.843, MSE: train=0.388, test=0.655 |
| 1013500 | M50  | Adaptive RSE | 365  | NSE: train=0.913, test=0.858, MSE: train=0.324, test=0.594 |
| 1013500 | M50  |     MSE      | 365  | NSE: train=0.913, test=0.858, MSE: train=0.323, test=0.597 |
| 1013500 | M100 |     MSE      | 365  | NSE: train=0.902, test=0.868, MSE: train=0.361, test=0.554 |
| 6431500 | M50  |     MSE      | 365  |   NSE: train=0.17, test=0.11, MSE: train=0.08, test=0.03   |

### 实验结论

1. 插值方法的不同对于模型预测精度影响不大
2. 使用batch training (loss:MAE)与完整序列输入的得到的精度差别较大, r2约0.3
3. time len 定义为选取batch的长度，按time len不同迭代的dataset长度也不同，当为all input时，time len为整个序列的长度
4. 在损失函数的选取上，不同损失函数所得到的预测精度有明显的差异，在使用RSE=1-NSE损失函数时，模型能够得到最优的预测性能，而使用其他损失函数下，最终的预测精度仍有一定的偏差
5. 使用Adaptive RSE方法已成功解决精度问题，以及RSE在batch训练中的不适用问题
6. 关于损失函数的思考Batch RSE，有点像不同样本下MSE的权重求和，也即是一种weight MSE，而weight的计算依据则是根据样本的偏差倒数得到，即偏差越大其所占权重越小，
而对于Adaptive RSE，这个的权重就变为了M0模型的预测偏差，预测偏差越大其所占权重越小
7. 内部模型的预训练是相当重要的，会很大程度提升模型的训练效率，同时提升预测精度
8. 在总体上将RSE作为优化目标实质上与MSE没有任何差异，因为其分母是一个固定值；而对于batch训练的话，RSE与MSE就有差异，


### todo
1. 在batch训练中将每一次的求解结果返回并保存S，然后更新每个batch的S初始值