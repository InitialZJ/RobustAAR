clc
clear
left=100;
right=3900;
load('error_data.mat')
z_pd_image=ez(1:length(ez))*1;
ex_image=ex(1:length(ez))/640;
ey_image=ey(1:length(ez))/360;

t=(1:length(ez))*0.1;

close all
h1=figure(1);
plot(t,ex_image,"LineWidth",2);
hold on
plot(t,ey_image,"LineWidth",2);
hold off

grid on
legend(["e_x","e_y"])
title("ImageError")
xlabel("t/s")
xlim([0 length(ez)*0.1])

z_pd=pz(1:length(pz));
x_pd=px(1:length(pz));
y_pd=py(1:length(pz));

h2=figure(2);
plot(t,z_pd_image,"LineWidth",2);
hold on
plot(t,x_pd,"LineWidth",2);
plot(t,y_pd,"LineWidth",2);
hold off

grid on
legend(["x^p_{dx}","x^p_{dy}","x^p_{dz}"])
title("DocingError")
xlabel("t/s")
xlim([0 length(ez)*0.1])

h3=figure(3);
plot(t,z_pd_image,"LineWidth",2,'Color','black');
grid on
legend("z^c_d")
title("DepthError")
xlabel("t/s")
xlim([0 length(ez)*0.1])

h4=figure(4);
set(figure(4),'Position',[0,0,700,340])

subplot(2,2,1);
plot(t,ex_image,"LineWidth",2,'Color','black','LineStyle','-');
hold on
plot(t,ey_image,"LineWidth",2,'Color','black','LineStyle','-.');
hold off
grid on
legend(["$e_x$","$e_y$"],'Interpreter', 'latex','FontSize',10,'Location','northeast')
% title("图像误差")
xlabel("$t$ (s)",'Interpreter', 'latex','FontSize',10)
ylabel("图像误差",'Interpreter', 'latex','FontSize',10)
xlim([0 length(ez)*0.1])

subplot(2,2,3);
plot(t,z_pd_image,"LineWidth",2,'Color','black');
grid on
legend("$z^{\rm{c}}_{\rm{d}}$",'Interpreter', 'latex','FontSize',10)
% title("深度误差")
xlabel("$t$ (s)",'Interpreter', 'latex','FontSize',10)
ylabel("深度误差(m)",'Interpreter', 'latex','FontSize',10)
% set(gca,"FontName",'Times New Roman','FontSize',10);
xlim([0 length(ez)*0.1])

subplot(1,2,2);
plot(t,z_pd_image,"LineWidth",2,'Color','black','LineStyle','-');
hold on
plot(t,x_pd,"LineWidth",2,'Color','black','LineStyle','-.');
plot(t,y_pd,"LineWidth",2,'Color','black','LineStyle',':');
hold off
grid on
legend(["$p^{\rm{p}}_{\rm{d},\textit{x}}$","$p^{\rm{p}}_{\rm{d},\textit{y}}$","$p^{\rm{p}}_{\rm{d},\textit{z}}$"],'Interpreter', 'latex','FontSize',10)
% title("对接误差")
xlabel("$t$ (s)",'Interpreter', 'latex','FontSize',10)
ylabel("对接位置误差(m)",'Interpreter', 'latex','FontSize',10)
% set(gca,"FontName",'Times New Roman','FontSize',10);
xlim([0 length(ez)*0.1])

figurename1='ImageError' ;
figurename2='DockingError' ;
figurename3='DepthError' ;
figurename4='Error' ;
savefig(h1,figurename1);
savefig(h2,figurename2);
savefig(h3,figurename3);
savefig(h4,figurename4);
