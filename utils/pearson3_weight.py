# Pearson-III curve plotting and fitting,
# used for hydrological analysis and hydraulic calculations.
# v6.1
# Copyright (c) 2020 -- 2021 ListLee

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import probscale
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import pearson3

USE_TEX = False

if USE_TEX:
    matplotlib.use("pgf")

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["pgf.rcfonts"] = False
    plt.rcParams["pgf.preamble"] += r"""
        \usepackage{xeCJK}
        \usepackage{amsmath}
        \usepackage{siunitx}
        \sisetup{detect-all}
        \usepackage{unicode-math}
        \setsansfont{FiraGO}
        \setmathfont{Fira Math}
        \setCJKsansfont{Source Han Sans SC}
    """
else:
    plt.rcParams["font.sans-serif"] = "SimHei"


class Data:
    """
    # 水文数据类
    ## 构造函数参数

    + `arr`：实测水文数据
    """

    def __init__(self, arr):
        self.arr = np.sort(arr)[::-1]
        # 降序排序输入数组
        self.n = len(arr)
        # 实测期长度
        self.extreme_num = 0
        # 特大洪水数

    def history(self, arr, length, num=0):
        """
        # 历史洪水数据

        ## 输入参数

        + `arr` 历史特大洪水序列，均为特大洪水
        + `length` 调查期长度

        + `num` 特大洪水数，包括历史特大洪水与实测特大洪水，默认为历史特大洪水数
        """
        self.historia = np.sort(arr)[::-1]
        # 历史洪水序列
        self.length = length
        # 调查期长度
        self.extreme_num = max(len(self.historia), num)
        # 特大洪水数
        self.extreme_num_in_measure = self.extreme_num - len(arr)
        # 实测期特大洪水数

        # 特大洪水序列与一般洪水序列
        self.extreme = self.historia
        self.ordinary = self.arr
        if self.extreme_num_in_measure > 0:
            for i in range(self.extreme_num_in_measure):
                self.extreme = np.append(self.extreme, self.arr[i])
            self.ordinary = np.delete(self.arr,
                                      range(self.extreme_num_in_measure))

        self.arr = np.sort(np.append(self.extreme, self.ordinary))[::-1]

    def figure(self, grid=True, logVert=False):
        """
        # 绘制图形

        ## 输入参数

        + `gird`：是否显示背景网格，默认为 `True`

        + `logVert`：纵坐标是否为对数坐标，默认为 `False`
        """
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        # 创建「画板」与「画布」

        self.ax.set_xscale("prob")
        # 横坐标改为概率坐标

        self.ax.set_xlabel(r"频率 $P$（%）")
        self.ax.set_ylabel(r"流量 $Q$（\si{m\cubed /s}）")

        self.ax.grid(grid)
        # 背景网格

        if logVert:
            self.ax.set_yscale("log")

    def empi_scatter(self, empi_prob=None):
        """
        # 点绘经验概率点
        """
        # 数学期望公式计算经验概率
        if empi_prob is None:
            if self.extreme_num == 0:
                self.empi_prob = (np.arange(self.n) + 1) / (self.n + 1) * 100
            else:
                self.extreme_prob = (np.arange(self.extreme_num) +
                                     1) / (self.length + 1) * 100
                self.ordinary_prob = self.extreme_prob[-1] + (
                        100 - self.extreme_prob[-1]) * (
                                             np.arange(self.n - self.extreme_num_in_measure) +
                                             1) / (self.n - self.extreme_num_in_measure + 1)
                self.empi_prob = np.append(self.extreme_prob,
                                           self.ordinary_prob)
        else:
            self.empi_prob = empi_prob

        # 画布坐标轴设置
        prob_lim = lambda prob: 1 if prob > 1 else 10 ** (np.ceil(
            np.log10(prob) - 1))

        self.prob_lim_left = prob_lim(self.empi_prob[0])
        self.prob_lim_right = 100 - prob_lim(100 - self.empi_prob[-1])
        self.ax.set_xlim(self.prob_lim_left, self.prob_lim_right)

        # 点绘经验概率
        if self.extreme_num:
            self.ax.scatter(self.ordinary_prob,
                            self.ordinary,
                            marker="o",
                            c="none",
                            edgecolors="k",
                            label="一般洪水经验概率点")
            self.ax.scatter(self.extreme_prob,
                            self.extreme,
                            marker="x",
                            c="k",
                            label="特大洪水经验概率点")
        else:
            self.ax.scatter(self.empi_prob,
                            self.arr,
                            marker="o",
                            c="none",
                            edgecolors="k",
                            label="经验概率点")

    def stat_params(self, output=True):
        """
        # 输出数据的统计参数

        ## 输入参数

        + `output`：是否在控制台输出参数，默认为 True
        """
        if self.extreme_num == 0:
            self.expectation = np.mean(self.arr)
            # 期望
            self.modulus_ratio = self.arr / self.expectation
            # 模比系数
            self.coeff_of_var = np.sqrt(
                np.sum((self.modulus_ratio - 1) ** 2) / (self.n - 1))
            # 变差系数

        else:
            self.expectation = (np.sum(self.extreme) +
                                (self.length - self.extreme_num) /
                                (self.n - self.extreme_num_in_measure) *
                                np.sum(self.ordinary)) / self.length
            self.coeff_of_var = (np.sqrt(
                (np.sum((self.extreme - self.expectation) ** 2) +
                 (self.length - self.extreme_num) /
                 (self.n - self.extreme_num_in_measure) * np.sum(
                            (self.ordinary - self.expectation) ** 2)) /
                (self.length - 1))) / self.expectation

        self.coeff_of_skew = stats.skew(self.arr, bias=False)
        # 偏态系数
        if output:
            print("期望 EX 为 %.2f" % self.expectation)
            print("变差系数 Cv 为 %.4f" % self.coeff_of_var)
            print("偏态系数 Cs 为 %.4f" % self.coeff_of_skew)

    def moment_plot(self):
        """
        # 绘制矩法估计参数理论概率曲线
        """
        x = np.linspace(self.prob_lim_left, self.prob_lim_right, 1000)
        theo_y = (pearson3.ppf(1 - x / 100, self.coeff_of_skew) *
                  self.coeff_of_var + 1) * self.expectation

        self.ax.plot(x, theo_y, "--", lw=1, label="矩法估计参数概率曲线")
        # 绘制理论曲线

    def plot_fitting(self, sv_ratio=0, ex_fitting=True, output=True):
        """
        # 优化适线

        ## 输入参数
        + `sv_ratio`：倍比系数，即偏态系数 `Cs` 与 变差系数 `Cv` 之比。

            默认为 0，即关闭倍比系数功能。

            - 当 `sv_ratio` ≠ 0 时，Cs 不参与适线运算中，且 `Cs` = `sv_ratio` × `Cv`；
            - 当 `sv_ratio` = 0 时，Cs 正常参与适线运算。
        + `ex_fitting`：适线时是否调整 EX，默认为 True
        + `output`：是否在控制台输出参数，默认为 True
        """

        if sv_ratio == 0:
            if ex_fitting:
                p3 = lambda prob, ex, cv, cs: (pearson3.ppf(
                    1 - prob / 100, cs) * cv + 1) * ex

                [self.fit_EX, self.fit_CV, self.fit_CS], pcov = curve_fit(
                    p3, self.empi_prob, self.arr,
                    [self.expectation, self.coeff_of_var, self.coeff_of_skew])

            else:
                p3 = lambda prob, cv, cs: (pearson3.ppf(1 - prob / 100, cs) *
                                           cv + 1) * self.expectation

                [self.fit_CV, self.fit_CS
                 ], pcov = curve_fit(p3, self.empi_prob, self.arr,
                                     [self.coeff_of_var, self.coeff_of_skew])

                self.fit_EX = self.expectation

        else:
            if ex_fitting:
                p3 = lambda prob, ex, cv: (pearson3.ppf(
                    1 - prob / 100, cv * sv_ratio) * cv + 1) * ex

                [self.fit_EX, self.fit_CV
                 ], pcov = curve_fit(p3, self.empi_prob, self.arr,
                                     [self.expectation, self.coeff_of_var])

            else:
                p3 = lambda prob, cv: (pearson3.ppf(
                    1 - prob / 100, cv * sv_ratio) * cv + 1) * self.expectation

                [self.fit_CV], pcov = curve_fit(p3, self.empi_prob, self.arr,
                                                [self.coeff_of_var])

                self.fit_EX = self.expectation

            self.fit_CS = self.fit_CV * sv_ratio

        if output:
            print("适线后")
            print("期望 EX 为 %.2f" % self.fit_EX)
            print("变差系数 Cv 为 %.4f" % self.fit_CV)
            print("偏态系数 Cs 为 %.4f" % self.fit_CS)

    def fitted_plot(self):
        """
        # 绘制适线后的概率曲线

        """

        x = np.linspace(self.prob_lim_left, self.prob_lim_right, 1000)
        theoY = (pearson3.ppf(1 - x / 100, self.fit_CS) * self.fit_CV +
                 1) * self.fit_EX

        self.ax.plot(x, theoY, lw=2, label="适线后概率曲线")
        # 绘制理论曲线

    def prob_to_value(self, prob):
        """
        # 由设计频率转换设计值

        ## 输入参数

        + `prob`：设计频率，单位百分数

        ## 输出参数

        + `value`：设计值
        """

        value = (pearson3.ppf(1 - prob / 100, self.fit_CS) * self.fit_CV +
                 1) * self.fit_EX

        print("%.4f%% 的设计频率对应的设计值为 %.2f" % (prob, value))

        return value

    def value_to_prob(self, value):
        """
        # 由设计值转换设计参数

        ## 输入参数

        + `value`：设计值

        ## 输出参数

        + `prob`：设计频率，单位百分数
        """
        prob = 100 - pearson3.cdf(
            (value / self.fit_EX - 1) / self.fit_CV, self.fit_CS) * 100

        print("%.2f 的设计值对应的设计频率为 %.4f%%" % (value, prob))

        return prob


def successive():
    data = Data(
        np.array([
            680.6, 468.4, 489.2, 450.6, 436.8, 586.2, 567.9, 473.9, 357.8,
            650.9, 391, 201.2, 452.4, 750.9, 585.2, 304.5, 370.5, 351, 294.8,
            360.9, 276, 549.1, 534, 349, 350, 372, 292, 485, 427, 620.8, 539,
            474, 292, 228, 357, 425, 365, 241, 267, 305, 306, 238.9, 277.3,
            170.8, 217.9, 208.5, 187.9
        ]))
    # 本例取自《工程水文学》（2010 年第 4 版，詹道江 徐向阳 陈元芳 主编）P150～151 表 6-3

    data.figure()
    data.empi_scatter()
    data.stat_params()
    data.moment_plot()
    data.plot_fitting()
    data.fitted_plot()

    data.ax.legend()

    data.fig.savefig("successive.svg", transparent=True)


def nonsuccessive():
    data = Data(
        np.array([
            1800, 530, 590, 1460, 2440, 490, 1060, 1790, 1480, 2770, 1420, 410,
            7100, 2200, 3400, 1300, 3080, 946, 430, 857, 421, 4500, 2800, 846,
            1400, 1100, 740, 3600, 1470, 690
        ]))
    data.history(np.array([9200]), 100, 2)
    # 本例取自《工程水文学》（1992 年第 2 版，王燕生 主编）P203 例 10-2

    data.figure()
    data.empi_scatter()
    data.stat_params()
    data.moment_plot()
    data.plot_fitting()
    data.fitted_plot()

    data.ax.legend()

    data.fig.savefig("nonsuccessive.pdf", transparent=True)


if __name__ == "__main__":
    successive()
    nonsuccessive()
