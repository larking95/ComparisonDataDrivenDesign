% FRITの設計スクリプト
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

% 参照モデル Td
tau = 0.02;
Td = c2d(1/(tau*s + 1), Ts);    % ゼロ次ホールドで離散化した一次遅れ系

% 制御器構造 C(rho)
Crho = @(rho) rho(1) + rho(2)*Ts/(1-z^-1) + rho(3)*(1-z^-1)/Ts; % PID制御器

% 初期制御器のパラメータ
rho0 = [10, 0.0, 0.0];              % 適当な比例制御器（安定化する）

% 初期制御器 C0
C0 = Crho(rho0);

%% 入出力データの取得
%データ取得に用いる入力信号 r  (ステップ信号)
N = 1000;           % データ数
r0 = ones([N, 1]);  % 信号のベクトル

% 制御対象モデル P
Tp = 0.74;
Kp = 1.02;
P = c2d(Kp/(Tp*s + 1), Ts);

% 初期の閉ループシステム
T0 = feedback(C0*P, 1);

% 制御対象の出力信号 y0
y0 = lsim(T0, r0);

% 制御入力信号 u0
u0 = lsim(C0, r0 - y0);

%% FRITによる設計
% 最適化問題の評価関数 f(x)
f = @(x) Jfrit(Crho(x), x, y0, u0, Td);

global C_ideal;     % 理想制御器
global fun;         % パラメトライズされた制御器

C_ideal = minreal(Td/(1 - Td)/P);
fun     = Crho;

% 最適なパラメータ rho
switch 3
    case 1
        % fminsearch を利用（MATLAB組込み）
        opt = optimset(...
            'Display', 'iter',...
            'PlotFcns', @optimplotfval,...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fminsearch(f, rho0, opt);
    case 2
        % fminunc を利用（Optimization toolbox）
        opt = optimoptions('fminunc',...
            'MaxIterations', 400,...
            'Display', 'iter',...
            'PlotFcn', 'optimplotfval',...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fminunc(f, rho0, opt);
    case 3
        % fmincon を利用（Optimization toolbox）
        opt = optimoptions('fmincon',...
            'Display', 'iter',...
            'PlotFcn', 'optimplotfval',...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fmincon(f, rho0,...
            [], [], [], [], zeros(size(rho0)), inf(size(rho0)), [], opt);
end

% 設計した制御器 C
C = Crho(rho);  % 制御器を求める

% 評価関数 Je
Je0 = Jfrit(C0, rho0, y0, u0, Td);      % 初期制御器の評価値
Je  = Jfrit(C, rho, y0, u0, Td);        % 設計した制御器の評価値
disp("J_frit(C_0) = " + num2str(Je0));
disp("J_frit(C_opt) = " + num2str(Je));

%% 性能の確認
% 制御器を実装したシステム全体 G
G = minreal(feedback(P*C, 1));

% % ステップ応答
% fig1 = figure('name', 'Step plot');
% stepplot(G, Td);

%ボード線図表示
fig2 = figure('name', 'Bode plot of controller');
bodeplot(G, Td, {1,100});

% ステップ応答
fig1 = figure('name', 'Step plot');
[yG, tG] = step(G);
[yd, td] = step(Td);
plot(td, yd, td, yd);
xlabel('Time [s]', 'Interpreter', 'latex');
ylabel('Velocity [m/s]', 'Interpreter', 'latex');
legend({'$$\frac{CP}{1+CP}$$', '$M$'},...
    'Interpreter', 'latex',... 
    'FontSize', 14,...
    'Location', 'southeast');

reshape_figure('qiita', [fig1, fig2]);

%   rhoの設計結果(VRFT)
%   35.358672621474007
%   47.814289705181935
%    0.000000000000000

%   rhoの設計結果(FRIT fmincon)
%   35.358436202831797
%   47.820263610360513
%    0.000339600315551

function val = Jfrit(C, rho, y0, u0, Td)
% FRIT の評価関数を計算する

% 引数の確認
validateattributes(C, {'tf', 'ss'}, {'scalar'}, 1);
validateattributes(rho, {'numeric'}, {'vector'}, 2);
validateattributes(y0, {'numeric'}, {'vector'}, 3);
validateattributes(u0, {'numeric'}, {'vector', 'size', size(y0)}, 4);
validateattributes(Td, {'tf', 'ss'}, {'scalar'}, 5);

% 安定性の確認(fmincon 以外の場合の対策)
if ~isempty(find(rho < 0, 1))
    val = inf;
    return;
end

% 疑似参照信号を計算
r_tilde = lsim(C^-1, u0) + y0;

% 参照モデルの応答を計算
y_tilde = lsim(Td, r_tilde);

% y0とy_tildeの差を評価
val = norm(y0 - y_tilde, 2)^2;
end

function stop = outfun(x, optimValues, state)
% 制御器が更新されていく様子をプロットする

% グローバル変数の宣言
global fun;
global C_ideal;

% 永続変数の定義
persistent fig;       % figure handle
persistent popt;

% 出力関数の中身
stop = false;           % 最適化を終了しない
if strcmp(state, 'init')
    fig = figure('name', 'How to optimize controller');
    popt = bodeoptions();
    popt.Xlim = [10^-2, 10^4];
    popt.Ylim = {[20, 80], [-90, 45]};
    T = timer('TimerFcn', "disp('init finished.')", 'StartDelay', 1);
    T.start();
    T.wait();
    T.delete();
    clearvars T;
end

% 現在の制御器
Cnow = fun(x);

% ボード線図のプロット
figure(fig);                      % 注目するfigureを変更
bodeplot(C_ideal, Cnow, popt);  % ボード線図

%テキストデータの作成、プロット
str = ['iter = ' num2str(optimValues.iteration) newline];
str = [str 'Kp = ' num2str(x(1), 3) newline]; % Pゲイン
str = [str 'Ki = ' num2str(x(2), 3) newline]; % Iゲイン
str = [str 'Kd = ' num2str(x(3), 3)];         % Dゲイン
text(0.02, 0.0, str, 'FontSize', 14);

drawnow
end
