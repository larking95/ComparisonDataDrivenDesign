% VRFTの設計スクリプト
% モータの速度制御系に対して設計する。

% Copyright (c) 2019 larking95(https://qiita.com/larking95)
% Released under the MIT Licence 
% https://opensource.org/licenses/mit-license.php

%% 初期化
clearvars;
close all;

%% 設計仕様の決定
% サンプリングタイムと演算子
Ts = 0.001;
s = tf('s');        % ラプラス演算子 s
z = tf('z', Ts);    % 時間進み演算子 z

% 参照モデル M
tau = 0.02;
M = c2d(1/(tau*s + 1), Ts); % ゼロ次ホールドで離散化した一次遅れ系

% 重み関数
gW = 100;
W = c2d(gW/(s + gW), Ts);   % ゼロ次ホールドで離散化した一次遅れ系

%制御器構造 beta
beta = minreal([1; Ts/(1 - z^-1); (1 - z^-1)/Ts]); % PID制御器

%% 入出力データの取得
%データ取得に用いる入力信号 u  (m系列信号)
n = 15;                                     % 段数
T = 2^n - 1;                                % 1周期当たりのデータ数
p = 15;                                     % データの周期数
N = T*p;                                    % データ数
u0 = idinput([T 1 p],'prbs',[0,1],[-1,1]);   % 信号のベクトル

% 入力信号のパワースペクトル密度 phi_u
phi_u = 1;                  %入力信号は白色性と仮定

% 制御対象モデル P
Tp = 0.74;
Kp = 1.02;
P = c2d(Kp/(Tp*s + 1), Ts);

% 出力信号 y0
y0 = lsim(P, u0);

%% VRFTによる設計
% プレフィルタ L
L = minreal(M*(1 - M)/phi_u);

% フィルタに通した入力信号 ul
ul = lsim(L, u0);

% 疑似誤差信号 el = 
el = lsim(L*(M^(-1) - 1), y0);

%パラメータ前の制御器出力 phi
phi = lsim(beta, el);

%最適なパラメータ rho
rho = phi\ul;               % 行列形式で最小二乗法を解く(mldivide)

%設計した制御器 C
C = minreal(rho.' * beta);  % 制御器を求める

%評価関数 Jmr
Jmr = mean(ul - phi * rho); % 行列形式で評価関数を確認

%% 性能の確認
% 制御器を実装したシステム全体 G
G = minreal(feedback(P*C, 1));

% ステップ応答
fig1 = figure('name', 'Step plot');
stepplot(G, M);

%ボード線図表示
fig2 = figure('name', 'Bode plot of controller');
bodeplot(G, M, {1,100});

% % ステップ応答
% fig1 = figure('name', 'Step plot');
% [yG, tG] = step(G);
% [yM, tM] = step(M);
% plot(tG, yG, tM, yM);
% xlabel('Time [s]', 'Interpreter', 'latex');
% ylabel('Velocity [m/s]', 'Interpreter', 'latex');
% legend({'$$\frac{CP}{1+CP}$$', '$M$'},...
%     'Interpreter', 'latex', 'FontSize', 14, 'Location', 'southeast');
% 
% reshape_figure('qiita', [fig1, fig2]);

%   rhoの設計結果
%   35.358672621474007
%   47.814289705181935
%    0.000000000000000


