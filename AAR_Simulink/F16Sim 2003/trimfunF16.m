%=====================================================
%      F16 nonlinear model trim cost function
%   for longitudinal motion, steady level flight
% (cost = sum of weighted squared state derivatives)
%
% Author: T. Keviczky
% Date:   April 29, 2002
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
%=====================================================
%%**********************************************%%  
%   Altered to work as a trimming function       %
%   for the  HIFI F_16 Model                     %
%%**********************************************%%

function [cost, Xdot, xu] = trimfunF16(UX0)

% Implementing limits:
% Thrust limits
global uu;
uu=UX0;
[t,x,y]=sim('F16trim',0);
Xdot=y';

% Create weight function
weight = [ 
            1            ...%V_dot
            1           ...%alpha_dpt
            ];

cost = weight*(Xdot.*Xdot);