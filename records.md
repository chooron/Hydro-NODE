# Hydrology ODE neural network and Loss Function

## Julia������Ԥ�⾫��
| ��վ���    | ģ��   | ��ʧ���� |                            ����                            |
|---------|------|:----:|:--------------------------------------------------------:|
| 1013500 | M100 | RSE  | NSE: train=0.948, test=0.897, MSE: train=0.19, test=0.43 |


## ��ǰ������

### Hydro-NODE����

1. ������������ʱ��������������ģ�Ͳ���Ѱ�Ź����൱�����������Ҫ��ǰѵ������ģ�ʹ���ģ����
2. solver��atol��rtol��������ģ��Ԥ�⾫����������Ӱ��(1e-3��1e-6)
3. ԭ�������д��ڹ��������

### Batch Training����

1. Batch training������Ҫȷ��ODE���̵ĳ�ʼֵ���⣬��ǰ����M0�����Ϊ׼����ʵ������ͬһ��ʧ�����£���ֵ����Ӱ�첻��
2. Batch training�޷�ʹ��RSE��ʧ��������Ҫ����ѡȡһ����ʧ�����õ���֮���ƽ������ǰʹ��MAE������ţ�����RSE�����0.3
3. Batch size��ѡȡ��һ���Ƚϴ�����⣬���ѡȡһ�����ʵ�batch size��û��һ����ȷ�Ķ���

**Hydro-NODE�������һ��ģ���������������Ƕ�����ģ��**

## Hydro-NODE����ʧ����ʵ��

### ʵ������

1. ��ʧ��������ѡȡ��

ʵ�����ã���վ��1013500��ģ�ͣ�M50�� �������torchdiffeq������Ϊȫ�������룬�Ż�����AdamW(lr=0.01), with schedular
�뷨��RSE�����н�y_mean�滻ΪM0��ʵ����������ѡ��������quantile loss


### ʵ���¼

| ��վ���    |   ��ʧ����   |                            ����                            |
|---------|:--------:|:--------------------------------------------------------:|
| 1013500 |   MAE    | NSE: train=0.865, test=0.830, MSE: train=0.60, test=0.71 |
| 1013500 | Huber(1) | NSE: train=0.85, test=0.83, MSE: train=0.556, test=0.71  |
| 1013500 | Huber(2) |  NSE: train=0.80, test=0.83, MSE: train=0.73, test=0.74  |
| 1013500 |   MSE    |  NSE: train=0.85, test=0.84, MSE: train=0.50, test=0.49  |
| 1013500 |   RSE    |  NSE: train=0.90, test=0.86, MSE: train=0.35, test=0.58  |
| 6431500 |   RSE    | NSE: train=-0.03, test=-0.68, MSE: train=0.10, test=0.06 |

2. ��ѵ��������Сѡȡ��

| ��վ���    | ģ��   |     ��ʧ����     | ������С |                             ����                             |
|---------|------|:------------:|------|:----------------------------------------------------------:|
| 1013500 | M50  |     MAE      | 365  | NSE: train=0.895, test=0.843, MSE: train=0.388, test=0.655 |
| 1013500 | M50  | Adaptive RSE | 365  | NSE: train=0.913, test=0.858, MSE: train=0.324, test=0.594 |
| 1013500 | M50  |     MSE      | 365  | NSE: train=0.913, test=0.858, MSE: train=0.323, test=0.597 |
| 1013500 | M100 |     MSE      | 365  | NSE: train=0.902, test=0.868, MSE: train=0.361, test=0.554 |
| 6431500 | M50  |     MSE      | 365  |   NSE: train=0.17, test=0.11, MSE: train=0.08, test=0.03   |

### ʵ�����

1. ��ֵ�����Ĳ�ͬ����ģ��Ԥ�⾫��Ӱ�첻��
2. ʹ��batch training (loss:MAE)��������������ĵõ��ľ��Ȳ��ϴ�, r2Լ0.3
3. time len ����Ϊѡȡbatch�ĳ��ȣ���time len��ͬ������dataset����Ҳ��ͬ����Ϊall inputʱ��time lenΪ�������еĳ���
4. ����ʧ������ѡȡ�ϣ���ͬ��ʧ�������õ���Ԥ�⾫�������ԵĲ��죬��ʹ��RSE=1-NSE��ʧ����ʱ��ģ���ܹ��õ����ŵ�Ԥ�����ܣ���ʹ��������ʧ�����£����յ�Ԥ�⾫������һ����ƫ��
5. ʹ��Adaptive RSE�����ѳɹ�����������⣬�Լ�RSE��batchѵ���еĲ���������
6. ������ʧ������˼��Batch RSE���е���ͬ������MSE��Ȩ����ͣ�Ҳ����һ��weight MSE����weight�ļ����������Ǹ���������ƫ����õ�����ƫ��Խ������ռȨ��ԽС��
������Adaptive RSE�������Ȩ�ؾͱ�Ϊ��M0ģ�͵�Ԥ��ƫ�Ԥ��ƫ��Խ������ռȨ��ԽС
7. �ڲ�ģ�͵�Ԥѵ�����൱��Ҫ�ģ���ܴ�̶�����ģ�͵�ѵ��Ч�ʣ�ͬʱ����Ԥ�⾫��
8. �������Ͻ�RSE��Ϊ�Ż�Ŀ��ʵ������MSEû���κβ��죬��Ϊ���ĸ��һ���̶�ֵ��������batchѵ���Ļ���RSE��MSE���в��죬


### todo
1. ��batchѵ���н�ÿһ�ε���������ز�����S��Ȼ�����ÿ��batch��S��ʼֵ