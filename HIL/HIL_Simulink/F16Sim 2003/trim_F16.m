function [trim_state, trim_thrust, trim_control, dLEF, xu] = trim_F16(thrust, elevator, alpha, ail, rud, vel, alt)
% 本函数用来做初始化，就是在不同高度和速度下，还有不同模式下（高低空），
% ...状态变量和控制量的初始位置
%
% trim_state = xu(1:12) 是飞机的12维状态空间的初值，分别是
%           北npos      东epos      高altitude
%           滚转角roll  俯仰角pitch  偏航角yaw
%           总速度Vt    迎角alpha    侧滑角beta
%           滚转角速度p 俯仰角速度q   偏航角速度r
%
% trim_thrust 就是 油门 初始位置
%
% trim_control 就是 升降舵、副翼、方向舵 的初始位置
%
% dLEF是襟翼的初始位置，高空模式在0~25°之间，低空模式就是0°


%================================================
%     F16 nonlinear model trimming routine
%  for longitudinal motion, steady level flight
%
% Author: T. Keviczky
% Date:   April 29, 2002
%
%
%      Added addtional functionality.
%      This trim function can now trim at three 
%      additional flight conditions
%         -  Steady Turning Flight given turn rate
%         -  Steady Pull-up flight - given pull-up rate
%         -  Steady Roll - given roll rate
%
% Coauthor: Richard S. Russell
% Date:     November 7th, 2002
%
%
%================================================

global altitude velocity fi_flag_Simulink
global phi psi p q r phi_weight theta_weight psi_weight

altitude = alt;
velocity = vel;
alpha = alpha*pi/180;  %convert to radians

% OUTPUTS: trimmed values for states and controls
% INPUTS:  guess values for thrust, elevator, alpha  (assuming steady level flight)

% Initial Guess for free parameters
UX0 = [thrust; elevator; alpha; ail; rud];  % free parameters: two control values & angle of attack

% Initialize some varibles
%
phi = 0; psi = 0;
p = 0; q = 0; r = 0;
phi_weight = 10; theta_weight = 10; psi_weight = 10;

disp('At what flight condition would you like to trim the F-16?');
disp('1.  Steady Wings-Level Flight.');
disp('2.  Steady Turning Flight.');
disp('3.  Steady Pull-Up Flight.');
disp('4.  Steady Roll Flight.');
FC_flag = input('Your Selection:  ');

switch FC_flag
    case 1
        % do nothing
    case 2
        r = input('Enter the turning rate (deg/s):  ');
        psi_weight = 0;
    case 3
        q = input('Enter the pull-up rate (deg/s):  ');
        theta_weight = 0;
    case 4    
        p = input('Enter the Roll rate    (deg/s):  ');
        phi_weight = 0;
    otherwise
        disp('Invalid Selection')
%        break;
end

% Initializing optimization options and running optimization:
OPTIONS = optimset('TolFun',1e-10,'TolX',1e-10,'MaxFunEvals',5e+04,'MaxIter',1e+04);

iter = 1;
while iter == 1
   
    [UX,FVAL,EXITFLAG,OUTPUT] = fminsearch('trimfun',UX0,OPTIONS);
    % UX0是初值，在trimfun函数下需找UX的最小值，作为当前高度和速度的稳态值
    % 这里计算的是，油门，各舵偏，以及迎角的初状态
    
    [cost, Xdot, xu] = trimfun(UX);
    % 再通过trimfun和得到的UX计算，初始输入xu就是系统输入，18个。
    % ？cost应该是用来做优化的，也就是上一条用的
    
    disp('Trim Values and Cost:');
    disp(['cost   = ' num2str(cost)])
    disp(['thrust = ' num2str(xu(13)) ' lb'])
    disp(['elev   = ' num2str(xu(14)) ' deg'])
    disp(['ail    = ' num2str(xu(15)) ' deg'])
    disp(['rud    = ' num2str(xu(16)) ' deg'])
    disp(['alpha  = ' num2str(xu(8)*180/pi) ' deg'])
    disp(['dLEF   = ' num2str(xu(17)) ' deg'])
    disp(['Vel.   = ' num2str(velocity) 'ft/s']) 
    flag = input('Continue trim rountine iterations? (y/n):  ','s'); 
    if flag == 'n'
        iter = 0;
    end
    UX0 = UX;
end

% For simulink:
trim_state=xu(1:12);
trim_thrust=UX(1);
trim_ele=UX(2);
trim_ail=UX(4);
trim_rud=UX(5);
trim_control=[UX(2);UX(4);UX(5)];
dLEF = xu(17);