function y = TankerVortexWindField(PosReceiver)
%#codegen


airDensity1 = 0.58950;               %查高度表所得
airSpeed1 = 120;

wingSpan1 = 40;
tankerWeight1 = 80000;               %KC-135最大载重量146000kg，取均值
gravity1 = 9.79434;                  %纬度43.3，高度7000计算所得

%见文献Flight Data Analysis and Simulation of Wind Effects During Aerial Refueling
vortexStrength = tankerWeight1*gravity1 / (airDensity1*airSpeed1*pi/4*wingSpan1);

%以下定义为自定义的气流坐标系

%翼尖位置，以机翼中心为原点
wingTipLeftPos = [0 -20 0]';
wingTipRightPos = [0 20 0]';

%尾翼尖所在位置，同上坐标系
tailTipLeftPos = [-12.44 -6.515 -1.27]';
tailTipRightPos = [-12.44 6.515 -1.27]';

%加油机质心（或其他参考点）在本坐标系中位置
tankerMassPos = [0 0 0]';

%输入为点在加油机气流坐标系位置，转化为在本自定义坐标系位置
receiverPos =PosReceiver + tankerMassPos;

%左侧(L)翼尖涡流计算
rVorL = sqrt((receiverPos(2)-wingTipLeftPos(2))^2 + (receiverPos(3)-wingTipLeftPos(3))^2);      %离涡线距离(左侧翼尖涡流)
tVorL = -receiverPos(1)/airSpeed1;                                                               %涡时间常数 
TVorL = vortexStrength;                                                                         %涡强度
vVorL = 0.06*TVorL;                                                                             %涡粘性
rVorCoreL = 2.24*sqrt(abs(vVorL*tVorL));                                                             %涡核
%rVorCoreL = 2.24*sqrt(vVorL*tVorL)

WVorL = TVorL*rVorL/(2*pi*(rVorL^2+rVorCoreL^2))*(1-exp(-rVorL^2/(4*vVorL*tVorL)));             %诱导风场大小   
thetaVorL = atan((receiverPos(3)-wingTipLeftPos(3))/(receiverPos(2)-wingTipLeftPos(2)));        %诱导风场角度，以oY为轴，向下旋转为正    
if thetaVorL < 0
   thetaVorL = thetaVorL + pi;
end

WVorLVector = [0 -WVorL*sin(thetaVorL) WVorL*cos(thetaVorL)]';                                  %涡场三维分解(自定义的气流系)


%右侧(R)翼尖涡流计算
rVorR = sqrt((receiverPos(2)-wingTipRightPos(2))^2 + (receiverPos(3)-wingTipRightPos(3))^2);
tVorR = -receiverPos(1)/airSpeed1;
TVorR = vortexStrength;
vVorR = 0.06*TVorR;
rVorCoreR = 2.24*sqrt(abs(vVorR*tVorR));

WVorR = TVorR*rVorR/(2*pi*(rVorR^2+rVorCoreR^2))*(1-exp(-rVorR^2/(4*vVorR*tVorR)));    
thetaVorR = atan((receiverPos(3)-wingTipRightPos(3))/(-receiverPos(2)+wingTipRightPos(2)));             %风场角度，以-oy为轴，向下为正    
if (-receiverPos(2)+wingTipRightPos(2)) < 0
   thetaVorR = thetaVorR + pi;
end

WVorRVector = [0 WVorR*sin(thetaVorR) WVorR*cos(thetaVorR)]';   


%整个机翼(Wing用字母W表示)等效涡流计算
rVorW = sqrt((receiverPos(1)-wingTipRightPos(1))^2 + (receiverPos(3)-wingTipRightPos(3))^2);            %到机翼轴线距离
tVorW = -receiverPos(1)/airSpeed1;

LVorLength = sqrt((receiverPos(1)-wingTipLeftPos(1))^2 + (receiverPos(2)-wingTipLeftPos(2)*pi/4)^2+ (receiverPos(3)-wingTipLeftPos(3))^2);       %到左翼尖距离
RVorLength = sqrt((receiverPos(1)-wingTipRightPos(1))^2 + (receiverPos(2)-wingTipRightPos(2)*pi/4)^2+ (receiverPos(3)-wingTipRightPos(3))^2);    %到右翼尖距离

gamaLCos = (receiverPos(2)-wingTipLeftPos(2)*pi/4)/LVorLength;                                          %到左侧翼尖角度余弦，oy为轴向下为正
gamaRCos = (-receiverPos(2)+wingTipRightPos(2)*pi/4)/RVorLength ;                                       %到右侧翼尖角度余弦，-oy为轴向下为正

TVorW = vortexStrength*(gamaLCos + gamaRCos)/2*4/pi;                                                    %机翼属于有限长度涡，用公式重新计算涡强度
vVorW = 0.06*TVorW;
rVorCoreW = 2.24*sqrt(abs(vVorW*tVorW));

WVorW = TVorW*rVorW/(2*pi*(rVorW^2+rVorCoreW^2))*(1-exp(-rVorW^2/(4*vVorW*tVorW)));         
thetaVorW = atan((receiverPos(3)-wingTipRightPos(3))/(-receiverPos(1)+wingTipRightPos(1)));              %风场角度，以-ox为轴，向下为正  
if (-receiverPos(1)+wingTipRightPos(1)) < 0
   thetaVorW = thetaVorW + pi;
end

WVorWVector = [WVorW*sin(thetaVorW), 0 , WVorW*cos(thetaVorW)]';   


%左侧(L)尾尖涡流计算
rVorLT = sqrt((receiverPos(2)-wingTipLeftPos(2))^2 + (receiverPos(3)-wingTipLeftPos(3))^2);      
tVorLT = -(receiverPos(1)-wingTipLeftPos(1))/airSpeed1;                                                                
TVorLT = vortexStrength/30;                                                                         
vVorLT = 0.06*TVorLT;                                                                             
rVorCoreLT = 2.24*sqrt(abs(vVorLT*tVorLT));                                                             

WVorLT = TVorLT*rVorLT/(2*pi*(rVorLT^2+rVorCoreLT^2))*(1-exp(-rVorLT^2/(4*vVorLT*tVorLT)));             %诱导风场大小   
thetaVorLT = atan((receiverPos(3)-wingTipLeftPos(3))/(receiverPos(2)-wingTipLeftPos(2)));               %诱导风场角度，以oY为轴，向下旋转为正    
if (receiverPos(2)-wingTipLeftPos(2)) < 0
   thetaVorLT = thetaVorLT + pi;
end

WVorLTVector = [0 WVorLT*sin(thetaVorLT) -WVorLT*cos(thetaVorLT)]';                                     %涡场三维分解(自定义的气流系)


%右侧(R)尾尖涡流计算
rVorRT = sqrt((receiverPos(2)-tailTipRightPos(2))^2 + (receiverPos(3)-tailTipRightPos(3))^2);
tVorRT = -(receiverPos(1)-wingTipLeftPos(1))/airSpeed1;
TVorRT = vortexStrength/30;
vVorRT = 0.06*TVorRT;
rVorCoreRT = 2.24*sqrt(abs(vVorRT*tVorRT));

WVorRT = TVorRT*rVorRT/(2*pi*(rVorRT^2+rVorCoreRT^2))*(1-exp(-rVorRT^2/(4*vVorRT*tVorRT)));    
thetaVorRT = atan((receiverPos(3)-tailTipRightPos(3))/(-receiverPos(2)+tailTipRightPos(2)));            %风场角度，以-oy为轴，向下为正    
if (-receiverPos(2)+tailTipRightPos(2)) < 0
   thetaVorRT = thetaVorRT + pi;
end

WVorRTVector = [0 -WVorRT*sin(thetaVorRT) -WVorRT*cos(thetaVorRT)]';   

%整个尾翼(Tail用字母T表示)等效涡流计算
rVorT = sqrt((receiverPos(1)-tailTipRightPos(1))^2 + (receiverPos(3)-tailTipRightPos(3))^2);            %到机翼轴线距离
tVorT = -(receiverPos(1)-wingTipLeftPos(1))/airSpeed1;

LVorLengthT = sqrt((receiverPos(1)-tailTipLeftPos(1))^2 + (receiverPos(2)-tailTipLeftPos(2)*pi/4)^2+ (receiverPos(3)-tailTipLeftPos(3))^2);       %到左翼尖距离
RVorLengthT = sqrt((receiverPos(1)-tailTipRightPos(1))^2 + (receiverPos(2)-tailTipRightPos(2)*pi/4)^2+ (receiverPos(3)-tailTipRightPos(3))^2);    %到右翼尖距离

gamaLCosT = (receiverPos(2)-tailTipLeftPos(2)*pi/4)/LVorLengthT;                                        %到左侧翼尖角度余弦，oy为轴向下为正
gamaRCosT = (-receiverPos(2)+tailTipRightPos(2)*pi/4)/RVorLengthT ;                                     %到右侧翼尖角度余弦，-oy为轴向下为正

TVorT = vortexStrength/30*(gamaLCosT + gamaRCosT)/2*4/pi;                                               %机翼属于有限长度涡，用公式重新计算涡强度
vVorT = 0.06*TVorT;
rVorCoreT = 2.24*sqrt(abs(vVorT*tVorT));

WVorT = TVorT*rVorT/(2*pi*(rVorT^2+rVorCoreT^2))*(1-exp(-rVorT^2/(4*vVorT*tVorT)));     
thetaVorT = atan((receiverPos(3)-tailTipRightPos(3))/(-receiverPos(1)+tailTipRightPos(1)));             %风场角度，以-ox为轴，向下为正  
if (-receiverPos(1)+tailTipRightPos(1)) < 0
   thetaVorT = thetaVorT + pi;
end

WVorTVector = [-WVorT*sin(thetaVorT), 0 , -WVorT*cos(thetaVorT)]';   


WTotalVector = WVorLVector + WVorRVector + WVorWVector + WVorLTVector + WVorRTVector + WVorTVector;

%输出风场与受油机机体轴对于
y = WTotalVector;