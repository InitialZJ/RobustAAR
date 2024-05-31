% mass = 636.94*14.59428;    %kg
% Inertia=[9496,0,-982.0;0,55814.0,0;-982.0,0,63100.0]*1.35583; %kg.m^2
% 
% S_ref=300.0*0.3048^2;%Reference area (m)
% b_ref=30.0*0.3048;%Reference span (m)
% d_ref=11.32*0.3048;%Reference length or mean chord length (m)
% 
% x_cg=0.35*11.32;%Center of Gravity
% x_ref=0.30*11.32;%Center if Pressure
% 
% beta=0;
% pqr=[0,0,0];
% altitude=3000;
% airSpeed=120;
fi_flag=0;
mass = 636.94;    %kg
Inertia=[9496,0,-982.0;0,55814.0,0;-982.0,0,63100.0]; %kg.m^2
aaa=0.115609526503456;
S_ref=300.0;%Reference area (m)
b_ref=30.0;%Reference span (m)
d_ref=11.32;%Reference length or mean chord length (m)

x_cg=0.35*11.32;%Center of Gravity
x_ref=0.30*11.32;%Center if Pressure

beta=0;
pqr=[0,0,0];
altitude=9843;
airSpeed=393.720000000000;
trim_state=[0;0;9843;0;aaa;0;393.720000000000;0.115609526503456;0;0;0;0];
xme_0=[0,0,-trim_state(3)];%Initial position in inertial axes [Xe,Ye,Ze]:
Vmw=[airSpeed*cos(aaa),0,airSpeed*sin(aaa)];%Initial velocity in body axes [U,v,w]:
Euler_0=[0,aaa,0];%Initial Euler orientation [roll, pitch, yaw]:
pm_0=[0,0,0];%Initial body rotation rates [p,q,r]:
%alpha0=trim_state(8)*180/pi;%Initialize LF_state with -alpha0 value in degrees

altitude=trim_state(3);%(ft)
velocity=trim_state(7);%(ft/s)

alpha0=trim_state(8)*180/pi;
if (fi_flag == 1)
    % Calculating qbar, ps and steady state leading edge flap deflection:
    % (see pg. 43 NASA report)
    rho0 = 2.377e-3; tfac = 1 - 0.703e-5*altitude;
    temp = 519*tfac; 
    if (altitude >= 35000) 
        temp = 390; 
    end;
    rho = rho0*tfac^4.14;
    qbar = 0.5*rho*velocity^2;
    ps = 1715*rho*temp;
    
    dLEF = 1.38*alpha0 - 9.05*qbar/ps + 1.45;
    
elseif (fi_flag== 0)
    dLEF = 0.0;
end

% Verify that the calculated leading edge flap
% have not been violated.
if (dLEF > 25)
    dLEF = 25;
elseif (dLEF < 0)
    dLEF = 0;
end;

trim_thrust=2279.08519954656;
trim_ele=-2.97973164860638;
trim_ail=0.00874607850103852;
trim_rud=0.0680302158182712;
%trim_control=[trim_thrust;trim_ele;trim_ail;trim_rud];
%trim_state=[0;0;9843;0;aaa;0;velocity*cos(aaa);0;velocity*sin(aaa);0;0;0];
trim_state1=[0;0;-9843;0;aaa;0;velocity*cos(aaa);0;velocity*sin(aaa);0;0;0];
out=[0;0;-9843;0;aaa;0;velocity;aaa;0;0;0;0];
trim_control=[trim_thrust/1000;trim_ele;trim_ail;trim_rud];
% Kp=[0	0	0	0	-9.646865162	0	0.261378724	-0.258820207	0	0	-5.345537529	0	0.000273926	0.021726033	0	0	0	0
% 0	0	0	0	-21.31023018	0	0.347608294	3.20634313	0	0	-12.94812208	0	0.000438866	0.05360637	0	0	0	0
% 0	0	0	-1.434909249	0	-1.369601197	0	0	-0.100164088	-0.743064144	0	-3.213700113	0	0	0.013108596	0.002455715	0	0
% 0	0	0	-0.295690905	0	-0.35240965	0	0	-0.227038694	-0.110802249	0	-1.225550863	0	0	0.002455715	0.001448419	0	0
% ];
% 
% Kp(:,7)=Kp(:,7)*1.5;





% linearized model : x_dot = A * x + B * u
%   
%   ----State Variables --------                          
%01      x = [ npos  ( ft )        - north position              
%02            epos  ( ft )        - east position    
%03            h     ( ft )        - height 
%04            phi   ( rad )       - Euler angle  (roll)   
%05            theta ( rad )       - Euler angle  (pitch)     
%06            psi   ( rad )       - Euler angle  (yaw)  
%07            Vt    ( ft/sec)     - total speed   
%08            alpha ( rad )       - angle of attack   
%09            beta  ( rad )       - sideslip angle 
%10            p     ( rad/sec )   - roll rate   
%11            q     ( rad/sec )   - pitch rate
%12            r     ( rad/sec )   - yaw rate              
%13            Thr   ( - )         - throttle actuator  
%14            Ele   ( - )         - elevon actuator    
%15            Ail   ( - )         - aileron actuator  
%16            Rud   ( - )         - rudder actuator  
%17            Q     ( ft )        - dynamic pressure  
%18            Ps    ( ft )        - static pressure   
%          ];            
%                                                         
%   ---- Control Variables --------                       
%      u = [ delta_t   ( lbs )     - throttle setting  1000~19000   
%            delta_e   ( deg )     - elevon deflection  +-30    
%            delta_a   ( deg )     - aileron deflection +-30    
%            delta_r   ( deg )     - rudder deflection  +-30
%           ];   

