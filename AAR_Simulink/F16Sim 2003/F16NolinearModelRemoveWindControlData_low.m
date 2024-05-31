
fi_flag=0;
mass = 636.94;    %kg
Inertia=[9496,0,-982.0;0,55814.0,0;-982.0,0,63100.0]; %kg.m^2

S_ref=300.0;%Reference area (m)
b_ref=30.0;%Reference span (m)
d_ref=11.32;%Reference length or mean chord length (m)

x_cg=0.35*11.32;%Center of Gravity
x_ref=0.30*11.32;%Center if Pressure

beta=0;
pqr=[0,0,0];
altitude=9843;
airSpeed=393.720000000000;
trimAlpha=0;
%trim_state=[0;0;altitude;0;trimAlpha;0;airSpeed;trimAlpha;0;0;0;0];
xme_0=[0,0,-9843];%Initial position in inertial axes [Xe,Ye,Ze]:
Vmw=[airSpeed*cos(trimAlpha),0,airSpeed*sin(trimAlpha)];%Initial velocity in body axes [U,v,w]:
Euler_0=[0,trimAlpha,0];%Initial Euler orientation [roll, pitch, yaw]:
pm_0=[0,0,0];%Initial body rotation rates [p,q,r]:



alpha0=trimAlpha*180/pi;
if (fi_flag == 1)
    % Calculating qbar, ps and steady state leading edge flap deflection:
    % (see pg. 43 NASA report)
    rho0 = 2.377e-3; tfac = 1 - 0.703e-5*altitude;
    temp = 519*tfac; 
    if (altitude >= 35000) 
        temp = 390; 
    end;
    rho = rho0*tfac^4.14;
    qbar = 0.5*rho*airSpeed^2;
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

trim_thrust=-206727.459731588;
trim_ele=-1.54942581309517;
trim_ail=-1.24450697103149e-10;
trim_rud=-4.77959844982516e-09;

trim_control=[trim_thrust;trim_ele;trim_ail;trim_rud];


trim_state=[0;0;0;0;trimAlpha;0;airSpeed;trimAlpha;0;0;0;0];
trim_state1=[trim_state;trim_control;0;0];


%%%simulink的一些转换矩阵
Kc=[
0.3048	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0.3048	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0.3048	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	57.29577951	0	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	57.29577951	0	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	57.29577951	0	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0.3048	0	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	57.29577951	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	57.29577951	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	57.29577951	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	57.29577951	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	57.29577951	0	0	0	0	0	0
0	0	1.47E-06	0	0.000124324	0	-0.000238787	0.185224164	0	0	0.008874503	0	4.88E-05	0.003173487	0	0	0	0
0	0	8.66E-16	-0.000124324	0	0	-1.41E-13	0	-2.296767309	-0.014355754	0	0.066891704	0	0	0.001957696	0.005745682	0	0
0	0	-5.53E-06	0	0	0	0.000898246	7.54488061	0	0	0.83270499	0	0	0.015232738	0	0	0	0
0	0	1.38E-06	0	0	0	0.000928168	0	0	0	0	0	0	0	0	0	0	0
0	0	-0.00428099	0	0	0	0.695487318	0	0	0	0	0	0	0	0	0	0	0
0	0	-0.056813516	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
];


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


A=[0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0
0	0	0	-4.48E-15	0	393.72	0	0	393.72	0	0	0	0	0	0	0	0	0
0	0	0	0	393.72	0	0	-393.72	0	0	0	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	1	0	1.14E-17	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0
0	0	4.73E-05	0	-32.17	0	-0.007682722	32.44020605	0	0	0.285528259	0	0.001570007	0.102103771	0	0	0	0
0	0	4.52E-07	0	0	0	-0.000244229	0.211639843	0	0	0.931953037	0	0	-0.001244788	0	0	0	0
0	0	7.08E-17	0.081707813	0	0	-5.75E-15	0	0.640505192	-0.001173123	0	-0.994533746	0	0	0.000159979	0.000469525	0	0
0	0	7.27E-15	0	0	0	-1.18E-12	8.47E-10	-11.45396544	-2.189309756	0	0.281552663	0	0	-0.322031376	0.057037949	0	0
0	0	-4.08E-07	0	0	0	6.63E-05	-0.741053502	0	0	-0.799376643	0	0	-0.083693788	0	0	0	0
0	0	-5.16E-15	0	0	0	8.39E-13	4.11E-12	4.271894673	0.007254683	0	-0.289140256	0	0	-0.020812325	-0.034017203	0	0
0	0	0	0	0	0	0	0	0	0	0	0	-1	0	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	-20.2	0	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	-20.2	0	0	0
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	-20.2	0	0
0	0	-4.70E-05	0	0	0	-0.031623327	1162.76729	0	0	0	0	0	0	0	0	-7.352941176	10.14705882
0	0	0	0	0	0	0	-415.3944015	0	0	0	0	0	0	0	0	0	-7.25
];

B=[0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
1	0	0	0
0	20.2	0	0
0	0	20.2	0
0	0	0	20.2
0	0	0	0
0	0	0	0
];

%%%输出一次为xd,h,yd
C=[1,zeros(1,17);
    0,0,1,zeros(1,15);
    0,1,zeros(1,16)];

C2m=[C(1,:);
     C(3,:);
     -C(2,:)
];

%%% change throttle input delta_t's range to 1~19
%%% to make sure four inputs have the same orders of magntiude
B(:,1)=B(:,1)*1000;



%%-------------------Longitudinal equation-------------------------------%%

% linearized model : x_lon_dot = A_lon * x_lon + B_lon * u_lon
%   
%   ----State Variables --------  
%   01.npos , 02.h , 03.theta , 04.Vt , 05.alpha , 06.q , 07.Thr , 08.Ele
%                                                         
%   ---- Control Variables --------  
%   delta_t delta_e
%
%   ---- Output --------  
%   C_lo_xd_h  -  npos , h
%

A_lon=A([1,3,5,7,8,11,13,14],[1,3,5,7,8,11,13,14]);
B_lon=B([1,3,5,7,8,11,13,14],1:2);
C_lon_xd_h=[1 zeros(1,7);
    0 1 zeros(1,6)];

%增广系统矩阵，加入h和xd的误差积分项
AA_lon=[A_lon,zeros(8,2);-C_lon_xd_h,zeros(2,2)];
BB_lon=[B_lon;zeros(2,2)];
Q_lon=diag([4 10 10 1 10 10 0 0 1 3]);

%%%R越大相应的输入量越小，控制越弱，响应时间越长
R_lon=diag([50 50]);
[k_lon,p_lon,e_lon]=lqr(AA_lon,BB_lon,Q_lon,R_lon);

%eig(AA_lon-BB_lon*k_lon)

Kp_lon=k_lon(:,1:8);
Ki_lon=-k_lon(:,9:10);


%%-------------------lateral equation-------------------------------%%

% linearized model : x_la_dot = A_la * x_la + B_la * u_la
%   
%   ----State Variables --------  
%   01.epos , 02.phi , 03.psi , 04.beta , 05.p , 06.r , 07.Ail , 08.Rud 
%                                                         
%   ---- Control Variables --------  
%   delta_a delta_r
%
%   ---- Output --------  
%   C_la_yd  -  epos
%

A_la=A([2,4,6,9,10,12,15,16],[2,4,6,9,10,12,15,16]);
B_la=B([2,4,6,9,10,12,15,16],3:4);
C_la_yd=[1 zeros(1,7)];

%增广系统矩阵，加入yd的误差积分项
AA_la=[A_la,zeros(8,1);-C_la_yd,zeros(1,1)];
BB_la=[B_la;zeros(1,2)];
Q_la=diag([10 2 4 4 2 2 0 0 4]);

%%%R越大相应的输入量越小，控制越弱，响应时间越长
R_la=diag([500 500]);
[k_la,p_la,e_la]=lqr(AA_la,BB_la,Q_la,R_la);

%eig(AA_la-BB_la*k_la)

Kp_la=k_la(:,1:8);
Ki_la=-k_la(:,9);


%%-------------------total equation-------------------------------%%
Kp=zeros(4,18);
Kp(1:2,[1,3,5,7,8,11,13,14])=Kp_lon;
Kp(3:4,[2,4,6,9,10,12,15,16])=Kp_la;

Kp(:,13)=zeros(4,1);

Ki=zeros(4,3);
Ki(1:2,1:2)=Ki_lon;
Ki(3:4,3)=Ki_la;
