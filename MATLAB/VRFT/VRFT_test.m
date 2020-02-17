% VRFT�̐݌v�X�N���v�g
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

% �Q�ƃ��f�� M
tau = 0.02;
M = c2d(1/(tau*s + 1), Ts); % �[�����z�[���h�ŗ��U�������ꎟ�x��n

% �d�݊֐�
gW = 100;
W = c2d(gW/(s + gW), Ts);   % �[�����z�[���h�ŗ��U�������ꎟ�x��n

%�����\�� beta
beta = minreal([1; Ts/(1 - z^-1); (1 - z^-1)/Ts]); % PID�����

%% ���o�̓f�[�^�̎擾
%�f�[�^�擾�ɗp������͐M�� u  (m�n��M��)
n = 15;                                     % �i��
T = 2^n - 1;                                % 1����������̃f�[�^��
p = 15;                                     % �f�[�^�̎�����
N = T*p;                                    % �f�[�^��
u0 = idinput([T 1 p],'prbs',[0,1],[-1,1]);   % �M���̃x�N�g��

% ���͐M���̃p���[�X�y�N�g�����x phi_u
phi_u = 1;                  %���͐M���͔��F���Ɖ���

% ����Ώۃ��f�� P
Tp = 0.74;
Kp = 1.02;
P = c2d(Kp/(Tp*s + 1), Ts);

% �o�͐M�� y0
y0 = lsim(P, u0);

%% VRFT�ɂ��݌v
% �v���t�B���^ L
L = minreal(M*(1 - M)/phi_u);

% �t�B���^�ɒʂ������͐M�� ul
ul = lsim(L, u0);

% �^���덷�M�� el = 
el = lsim(L*(M^(-1) - 1), y0);

%�p�����[�^�O�̐����o�� phi
phi = lsim(beta, el);

%�œK�ȃp�����[�^ rho
rho = phi\ul;               % �s��`���ōŏ����@������(mldivide)

%�݌v��������� C
C = minreal(rho.' * beta);  % ���������߂�

%�]���֐� Jmr
Jmr = mean(ul - phi * rho); % �s��`���ŕ]���֐����m�F

%% ���\�̊m�F
% ���������������V�X�e���S�� G
G = minreal(feedback(P*C, 1));

% �X�e�b�v����
fig1 = figure('name', 'Step plot');
stepplot(G, M);

%�{�[�h���}�\��
fig2 = figure('name', 'Bode plot of controller');
bodeplot(G, M, {1,100});

% % �X�e�b�v����
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

%   rho�̐݌v����
%   35.358672621474007
%   47.814289705181935
%    0.000000000000000


