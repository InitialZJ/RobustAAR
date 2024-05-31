function [trim_control,alpha] = trim_F16_test()
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

fcn='F16trim';
load_system(fcn);

UX0 = [0; 0; 0; 0; 0];  % free parameters: two control values & angle of attack

% Initializing optimization options and running optimization:
OPTIONS = optimset('TolFun',1e-10,'TolX',1e-10,'MaxFunEvals',5e+04,'MaxIter',1e+04);

try
    % Pre-compile the model
    feval(fcn, [], [], [], 'lincompile');
    iter = 1;
    while iter == 1

        % UX0是初值，在trimfun函数下需找UX的最小值，作为当前高度和速度的稳态值
        % 这里计算的是，油门，各舵偏，以及迎角的初状态
        [UX,FVAL,EXITFLAG,OUTPUT] = fminsearch('trimfun_F16',UX0,OPTIONS);
        [cost, Xdot, xu] = trimfun_F16(UX);
        disp('Trim Values and Cost:');
        disp(['cost   = ' num2str(cost)]);
        % 再通过trimfun和得到的UX计算，初始输入xu就是系统输入，18个。
        % ？cost应该是用来做优化的，也就是上一条用的
        Xdot
        xu
        flag = input('Continue trim rountine iterations? (y/n):  ','s'); 
        if flag == 'n'
            iter = 0;
        end
        UX0 = UX;
    end

catch Ex
    % Clean up 
    feval(fcn, [], [], [], 'term');
    rethrow(Ex)
end
feval(fcn, [], [], [], 'term');

trim_control=UX0(1:4);
alpha=UX0(5);

end
