%path(path,'F16Sim 2003/');

fi_flag = 1;  %选择高保真或低保真气动参数

mass = 636.94*14.59428;    %kg
Inertia=[9496,0,-982.0;0,55814.0,0;-982.0,0,63100.0]*1.35583; %kg.m^2


trim_state=[0;0;9843;0;0.115609526503456;0;393.720000000000;0.115609526503456;0;0;0;0];

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
xme_0=[0,0,-trim_state(3)*0.3048];%Initial position in inertial axes [Xe,Ye,Ze]:
Vmw=[trim_state(7)*0.3048*cos(trim_state(8)),0,trim_state(7)*0.3048*sin(trim_state(8))];%Initial velocity in body axes [U,v,w]:
Euler_0=[0,trim_state(8),0];%Initial Euler orientation [roll, pitch, yaw]:
pm_0=[0,0,0];%Initial body rotation rates [p,q,r]:

S_ref=27.870912;%Reference area (m)
b_ref=9.144;%Reference span (m)
d_ref=3.450336;%Reference length or mean chord length (m)

x_cg=0.09144;%Center of Gravity
x_ref=0.10668;%Center if Pressure
alpha0=trim_state(8)*180/pi;%Initialize LF_state with -alpha0 value in degrees


altitude=trim_state(3);%(ft)
velocity=trim_state(7);%(ft/s)
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
trim_control=[trim_thrust,trim_ele,trim_ail,trim_rud];
trim_state1=[0;0;0;0;0.115609526503456;0;393.720000000000;0.115609526503456;0;0;0;0;zeros(6,1)];