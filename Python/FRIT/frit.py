# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:50:23 2020

@author: syotaro
"""

import matplotlib.pyplot as plt
import numpy as np
import control.matlab as ctl

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, BFGS

def Jfrit(C, rho, y0, u0, Td):
    # FRIT の評価関数を計算する
    
    # 疑似参照信号を計算
    e_tilde, _, _ = ctl.lsim(C**-1, u0.flatten())
    r_tilde = e_tilde + y0
    
    # 参照モデルの応答を計算
    y_tilde, _, _ = ctl.lsim(Td, r_tilde.flatten())
    
    # y0とy_tildeの差を評価
    return np.linalg.norm(y0 - y_tilde, 2)**2

def callbackFunc(xk, state):
    callbackFunc.epoch += 1
    print("{}: Jfrit = {:4.2f}, rho = {}".format(
        callbackFunc.epoch, 
        state["fun"], 
        np.round(xk, 2)))
    callbackFunc.ax1.scatter(callbackFunc.epoch, state["fun"])
    if callbackFunc.epoch % 2 == 0:
        plt.pause(.01)
    
callbackFunc.epoch = 0
callbackFunc.fig1 = plt.figure()
callbackFunc.ax1 = callbackFunc.fig1.add_subplot(1, 1, 1)


# === 設計仕様の決定 ===
# サンプリングタイムと演算子
Ts = 0.001
s = ctl.tf([1,0], [1])         # ラプラス演算子 s
z = ctl.tf([1, 0], [1], Ts)    # 時間進み演算子 z

# 参照モデル Td
tau = 0.02
Td = ctl.c2d(1/(tau*s + 1), Ts)    # ゼロ次ホールドで離散化した一次遅れ系

# 制御器構造 C(rho)
Crho = lambda rho: rho[0] + rho[1]*Ts/(1-z**-1) + rho[2]*(1-z**-1)/Ts # PID制御器

# 初期制御器のパラメータ
rho0 = np.array([10, 0.0, 0.0])              # 適当な比例制御器（安定化する）

# 初期制御器 C0
C0 = Crho(rho0)

# === 入出力データの取得 ===
# データ取得に用いる入力信号 r (ステップ信号)
N  = 1000           # データ数
r0 = np.ones(N)     # 信号のベクトル

# 制御対象モデル P
Tp = 0.74
Kp = 1.02
P = ctl.c2d(Kp/(Tp*s + 1), Ts)

# 初期の閉ループシステム
T0 = ctl.feedback(C0*P, 1)

# 制御対象の出力信号 y0
y0, t0, _ = ctl.lsim(T0, r0)

# 制御入力信号 u0
u0, _, _ = ctl.lsim(C0, r0 - y0.flatten())

# === FRITによる設計 ===
# 最適化問題の評価関数 f(x)
f = lambda x: Jfrit(Crho(x), x, y0, u0, Td)
C_ideal = ctl.minreal(Td/(1 - Td)/P)

# 制約条件（すべてのパラメータが正）
cons = LinearConstraint(
    np.eye(rho0.size), 
    np.zeros_like(rho0), 
    np.full_like(rho0, np.inf))

# 最適なパラメータ rho
optResult = minimize(f, rho0,
                     method="trust-constr", 
                     jac="2-point",          # 勾配関数
                     hess=BFGS(),            # ヘシアンの推定方法
                     constraints=cons,
                     options={"maxiter":np.inf, "disp":True},
                     callback=callbackFunc
                     )
rho = optResult["x"]
# 設計した制御器 C
C = Crho(rho)  # 制御器を求める

# 評価値
print("初期の評価値\t： {:5.3f}".format(f(rho0)))
print("設計後の評価値\t： {:5.3f}".format(f(rho)))

# === 性能の確認 ===
# 制御器を実装したシステム全体 G
G = ctl.minreal(ctl.feedback(P*C, 1), verbose=False)

# ステップ応答
plt.figure()
plt.title("Step response of closed loop system")
timeRange = np.arange(0, 0.12 + Ts, Ts)
ym, _ = ctl.step(Td, timeRange)
yg, _ = ctl.step(G, timeRange)
plt.plot(timeRange, ym, timeRange, yg)
plt.xlabel("Time [s]")
plt.ylabel("Velocity [V]")
plt.legend(['Reference model', 'Closed loop system'], loc='lower right')

# ボード線図表示
plt.figure()
plt.title("Bode plot of closed loop system")
ctl.bode(Td, G)
plt.legend(['Reference model', 'Closed loop system'], loc='lower left')

plt.show()
