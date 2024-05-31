%path(path,'F16Sim 2003/');

fi_flag = 0;  %选择高保真或低保真气动参数

mass = 636.94*14.59428;    %kg
Inertia=[9496,0,-982.0;0,55814.0,0;-982.0,0,63100.0]*1.35583; %kg.m^2


trim_state=[0;0;-9843;0;0.115609526503456;0;393.720000000000;0.115609526503456;0;0;0;0];

% %01      x = [ npos  ( ft )        - north position              
% %02            epos  ( ft )        - east position    
% %03            -h     ( ft )        - height 
% %04            phi   ( rad )       - Euler angle  (roll)   
% %05            theta ( rad )       - Euler angle  (pitch)     
% %06            psi   ( rad )       - Euler angle  (yaw)  
% %07            u    ( ft/sec)     - total speed   
% %08            v ( rad )       - angle of attack   
% %09            w  ( rad )       - sideslip angle 
% %10            p     ( rad/sec )   - roll rate   
% %11            q     ( rad/sec )   - pitch rate
% %12            r     ( rad/sec )   - yaw rate              
% %13            Thr   ( - )         - throttle actuator  
% %14            Ele   ( - )         - elevon actuator    
% %15            Ail   ( - )         - aileron actuator  
% %16            Rud   ( - )         - rudder actuator  
% %17            Lef     ( ft )        - dynamic pressure  
% %18            Dlef    ( ft )        - static pressure   
% %          ];            
xme_0=[0,0,0];%Initial position in inertial axes [Xe,Ye,Ze]:
Vmw=[120*cos(6/180*pi),0,120*sin(6/180*pi)];%Initial velocity in body axes [U,v,w]:
Euler_0=[0,6/180*pi,0];%Initial Euler orientation [roll, pitch, yaw]:
pm_0=[0,0,0];%Initial body rotation rates [p,q,r]:

iy=[3,7,9,10,11,12];
%ix=[1,2,3,4,6,8,10,11,12,15,16];

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

trim_thrust=500;
trim_ele=-2.97973164860638;
trim_ail=0.00874607850103852;
trim_rud=0.0680302158182712;
trim_control=[trim_thrust;trim_ele;trim_ail;trim_rud];
trim_state1=[0;0;0;0;0.115609526503456;0;393.720000000000;0.115609526503456;0;0;0;0;zeros(6,1)];
control=[trim_thrust;trim_ele;0;0];
out=zeros(6,1);
iy=[1,2,3,4,5,6];
ix=[4,6,8,10,11,12,15,16];
%state=[0;0;-3000;0;6/180*pi;0;120*cos(trim_state(8));0;120*sin(trim_state(8));0;0;0;trim_thrust;trim_ele;0;0;dLEF;-alpha0];
state=[0;0;0;0;6/180*pi;0;120*cos(trim_state(8));0;120*sin(trim_state(8));0;0;0]
[x,u,y,dx]=trim('F16LinmodTest',[],control,out,[],[3,4],iy)
%[x,u,y,dx]=trim('F16LinmodTest')