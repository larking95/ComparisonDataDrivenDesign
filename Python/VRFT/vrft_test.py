# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:45:30 2019

@author: larking95
"""

import matplotlib.pyplot as plt
import numpy as np
import control.matlab as ctl

# from scipy.signal import max_len_seq
from scipy.signal import abcd_normalize


def prbs(n, p):
    # matlab compatible PRBS signal generator

    taps = {3: [0, -1], 4: [0, -1], 5: [1, -1], 6: [0, -1], 7: [0, -1],
            8: [0, 1, 6, -1], 9: [3, -1], 10: [2, -1], 11: [8, -1],
            12: [5, 7, 10, -1], 13: [8, 9, 11, -1], 14: [3, 7, 12, -1],
            15: [13, -1], 16: [3, 12, 14, -1], 17: [13, -1], 18: [10, -1]}
    N = (2**n - 1)*p
    x = np.ones(n, dtype=np.int8)
    u = np.zeros(N, dtype=np.int8)
    tap = taps[n]
    for i in range(N):
        u[i] = x[-1]
        x0 = x[tap[0]] ^ x[tap[1]]
        x = np.roll(x, 1)
        x[0] = x0
    return u


# === 設計仕様の決定 ===
# サンプリング時間
Ts = 0.001

# 参照モデル M
tau = 0.02
M = ctl.c2d(ctl.tf([1], [tau, 1]), Ts)  # ゼロ次ホールドで離散化した1次遅れ系

# 重み関数 W
gW = 100
W = ctl.c2d(ctl.tf([gW], [1, gW]), Ts)  # ゼロ次ホールドで離散化した1次遅れ系

# 制御器構造 beta
A, B, C, D = abcd_normalize(
    [[1., 0], [0, 0]],
    [[1.], [-1.]],
    [[0, 0], [Ts, 0], [0, -1/Ts]],
    [[1.], [0], [1/Ts]])
beta = ctl.ss(A, B, C, D, Ts)
del A, B, C, D

# === 入出力データの取得 ===
# データ取得に用いる入力信号 u  (m系列信号)
n = 15
T = 2**n - 1
p = 15
N = T*p
# u0, _ = max_len_seq(n, length=N)  # MATLABとは違う系列の信号となってしまう
u0 = prbs(n, p)
u0 = -(2.*u0 - 1.)

# 入力信号のパワースペクトル密度 phi_u
phi_u = 1.

# 制御対象モデル P
Tp = 0.74
Kp = 1.02
P = ctl.c2d(ctl.tf([Kp], [Tp, 1]), Ts)

# 出力信号 y0
y0, t0, _ = ctl.lsim(P, u0)

# === VRFTによる設計 ===
# プレフィルタ L
L = ctl.minreal(M*(1 - M)/phi_u)

# フィルタに通した入力信号 ul
ul, _, _ = ctl.lsim(L, u0)

# 疑似誤差信号 el
el, _, _ = ctl.lsim(ctl.minreal(L*(M**-1 - 1)), y0.flatten())

# パラメータ前の制御器出力
phi, _, _ = ctl.lsim(beta, el.flatten())

# 最適なパラメータ rho
solution = np.linalg.lstsq(phi, ul, rcond=None)
rho = solution[0]

# 設計した制御器 C
C = ctl.ss([0], [0, 0, 0], [0], rho.T, Ts) * beta   # 制御器を求める

# 評価関数 Jmr
Jmr = np.mean(ul - np.dot(phi, rho))    # 行列形式で評価関数を確認

# === 性能の確認 ===
# 制御器を実装したシステム全体 G
G = ctl.minreal(ctl.feedback(P*C, 1))

# ステップ応答
plt.figure()
plt.title("Step response of closed loop system")
timeRange = np.arange(0, 0.12 + Ts, Ts)
ym, _ = ctl.step(M, timeRange)
yg, _ = ctl.step(G, timeRange)
plt.plot(timeRange, ym, timeRange, yg)
plt.xlabel("Time [s]", usetex=True)
plt.ylabel("Velocity [V]")
plt.legend(['Reference model', 'Closed loop system'], loc='lower right')

# ボード線図表示
plt.figure()
plt.title("Bode plot of closed loop system")
ctl.bode(M, G)
plt.legend(['Reference model', 'Closed loop system'], loc='lower left')


plt.show()
