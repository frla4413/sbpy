clear all
close all
clc
save_dir = "/home/fredrik/work/abl/abl_report/img/mms_figures/";
fig_name = "mms_N11_sbp21_no_pen_robin_error_u.png";
file_name = "data_files/time_independent_mms/mms_time_indep_N11_sbp21_no_pen_robin_err_u.dat";
title_str = "No penalty terms";
data = load(file_name);
N = 11; 

X = reshape(data(:,1),N,N);
Y = reshape(data(:,2),N,N);
U = reshape(data(:,3),N,N);
V = reshape(data(:,4),N,N);

surf(X,Y,U)
shading interp 
colormap jet
set(gca, 'fontsize',15)
xlabel('x')
ylabel('y')
xticks([0 0.5 1])
yticks([1 1.5 2])
zticks([0 0.5 1])
xlim([0 1])
ylim([1 2])
zlim([0 1])
title(title_str)
%caxis([0 0.5073])
%exportgraphics(gca, save_dir + fig_name);

