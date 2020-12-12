% FRIT�̐݌v�X�N���v�g
% ���[�^�̑��x����n�ɑ΂��Đ݌v����B

% Copyright (c) 2019 larking95(https://qiita.com/larking95)
% Released under the MIT Licence
% https://opensource.org/licenses/mit-license.php

%% ������
clearvars;
close all;

%% �݌v�d�l�̌���
% �T���v�����O�^�C���Ɖ��Z�q
Ts = 0.001;
s = tf('s');        % ���v���X���Z�q s
z = tf('z', Ts);    % ���Ԑi�݉��Z�q z

% �Q�ƃ��f�� Td
tau = 0.02;
Td = c2d(1/(tau*s + 1), Ts);    % �[�����z�[���h�ŗ��U�������ꎟ�x��n

% �����\�� C(rho)
Crho = @(rho) rho(1) + rho(2)*Ts/(1-z^-1) + rho(3)*(1-z^-1)/Ts; % PID�����

% ���������̃p�����[�^
rho0 = [10, 0.0, 0.0];              % �K���Ȕ�ᐧ���i���艻����j

% ��������� C0
C0 = Crho(rho0);

%% ���o�̓f�[�^�̎擾
%�f�[�^�擾�ɗp������͐M�� r  (�X�e�b�v�M��)
N = 1000;           % �f�[�^��
r0 = ones([N, 1]);  % �M���̃x�N�g��

% ����Ώۃ��f�� P
Tp = 0.74;
Kp = 1.02;
P = c2d(Kp/(Tp*s + 1), Ts);

% �����̕��[�v�V�X�e��
T0 = feedback(C0*P, 1);

% ����Ώۂ̏o�͐M�� y0
y0 = lsim(T0, r0);

% ������͐M�� u0
u0 = lsim(C0, r0 - y0);

%% FRIT�ɂ��݌v
% �œK�����̕]���֐� f(x)
f = @(x) Jfrit(Crho(x), x, y0, u0, Td);

global C_ideal;     % ���z�����
global fun;         % �p�����g���C�Y���ꂽ�����

C_ideal = minreal(Td/(1 - Td)/P);
fun     = Crho;

% �œK�ȃp�����[�^ rho
switch 3
    case 1
        % fminsearch �𗘗p�iMATLAB�g���݁j
        opt = optimset(...
            'Display', 'iter',...
            'PlotFcns', @optimplotfval,...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fminsearch(f, rho0, opt);
    case 2
        % fminunc �𗘗p�iOptimization toolbox�j
        opt = optimoptions('fminunc',...
            'MaxIterations', 400,...
            'Display', 'iter',...
            'PlotFcn', 'optimplotfval',...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fminunc(f, rho0, opt);
    case 3
        % fmincon �𗘗p�iOptimization toolbox�j
        opt = optimoptions('fmincon',...
            'Display', 'iter',...
            'PlotFcn', 'optimplotfval',...
            'OutputFcn', @outfun);
        [rho, ~, ~, info] = fmincon(f, rho0,...
            [], [], [], [], zeros(size(rho0)), inf(size(rho0)), [], opt);
end

% �݌v��������� C
C = Crho(rho);  % ���������߂�

% �]���֐� Je
Je0 = Jfrit(C0, rho0, y0, u0, Td);      % ���������̕]���l
Je  = Jfrit(C, rho, y0, u0, Td);        % �݌v���������̕]���l
disp("J_frit(C_0) = " + num2str(Je0));
disp("J_frit(C_opt) = " + num2str(Je));

%% ���\�̊m�F
% ���������������V�X�e���S�� G
G = minreal(feedback(P*C, 1));

% % �X�e�b�v����
% fig1 = figure('name', 'Step plot');
% stepplot(G, Td);

%�{�[�h���}�\��
fig2 = figure('name', 'Bode plot of controller');
bodeplot(G, Td, {1,100});

% �X�e�b�v����
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

%   rho�̐݌v����(VRFT)
%   35.358672621474007
%   47.814289705181935
%    0.000000000000000

%   rho�̐݌v����(FRIT fmincon)
%   35.358436202831797
%   47.820263610360513
%    0.000339600315551

function val = Jfrit(C, rho, y0, u0, Td)
% FRIT �̕]���֐����v�Z����

% �����̊m�F
validateattributes(C, {'tf', 'ss'}, {'scalar'}, 1);
validateattributes(rho, {'numeric'}, {'vector'}, 2);
validateattributes(y0, {'numeric'}, {'vector'}, 3);
validateattributes(u0, {'numeric'}, {'vector', 'size', size(y0)}, 4);
validateattributes(Td, {'tf', 'ss'}, {'scalar'}, 5);

% ���萫�̊m�F(fmincon �ȊO�̏ꍇ�̑΍�)
if ~isempty(find(rho < 0, 1))
    val = inf;
    return;
end

% �^���Q�ƐM�����v�Z
r_tilde = lsim(C^-1, u0) + y0;

% �Q�ƃ��f���̉������v�Z
y_tilde = lsim(Td, r_tilde);

% y0��y_tilde�̍���]��
val = norm(y0 - y_tilde, 2)^2;
end

function stop = outfun(x, optimValues, state)
% ����킪�X�V����Ă����l�q���v���b�g����

% �O���[�o���ϐ��̐錾
global fun;
global C_ideal;

% �i���ϐ��̒�`
persistent fig;       % figure handle
persistent popt;

% �o�͊֐��̒��g
stop = false;           % �œK�����I�����Ȃ�
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

% ���݂̐����
Cnow = fun(x);

% �{�[�h���}�̃v���b�g
figure(fig);                      % ���ڂ���figure��ύX
bodeplot(C_ideal, Cnow, popt);  % �{�[�h���}

%�e�L�X�g�f�[�^�̍쐬�A�v���b�g
str = ['iter = ' num2str(optimValues.iteration) newline];
str = [str 'Kp = ' num2str(x(1), 3) newline]; % P�Q�C��
str = [str 'Ki = ' num2str(x(2), 3) newline]; % I�Q�C��
str = [str 'Kd = ' num2str(x(3), 3)];         % D�Q�C��
text(0.02, 0.0, str, 'FontSize', 14);

drawnow
end
