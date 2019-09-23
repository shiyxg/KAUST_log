clear all;close all;clc;

addpath('./core')


model_dir = 'C_model/';

% % load the initial velocity model
 load(['./model/',model_dir,'C_model.mat']);
% load(['./model/',model_dir,'acquisition.mat']);
% load(['./model/',model_dir,'v0.mat']); 
[nz,nx]=size(vs);
dx=1;
iter = 20;
result_dir = ['./result/',model_dir];


vs_inv= read_bin([result_dir,'Vs_inv_',num2str(iter),'.bin'],nz,nx);
vs_initial = read_bin([result_dir,'vs_initial.bin'],nz,nx);
residual= read_bin([result_dir,'data_misfit.bin'],1,iter);
residual_m = read_bin([result_dir,'model_misfit.bin'],1,iter);
xx = 0+(0:nx-1)*dx; zz = (0:nz-1)*dx;


ascal1=719;
ascal2=992;
t_x = -15;
t_y = -15;
figure;
subplot(311)
imagesc(xx,zz,vs)
caxis([ascal1 ascal2])
ylabel('Depth(m)','fontsize',12)
title('True S Velocity Model','fontsize',12)
axis image
colormap(jet);
h=colorbar;
set(gca,'fontsize',12);
set(get(h,'Title'),'String','m/s','fontsize',12)
text(t_x,t_y,'(a)','fontsize',12);
subplot(312)
imagesc(xx,zz,vs_initial);
caxis([ascal1 ascal2])
ylabel('Depth(m)','fontsize',12)
% xlabel('X Distance (m)','fontsize',12)
title('S Velocity Initial Model','fontsize',12)
axis image
colormap(jet);
h=colorbar;
set(get(h,'Title'),'String','m/s','fontsize',12)
text(t_x,t_y,'(b)','fontsize',12);
set(gca,'fontsize',12)
subplot(313)
imagesc(xx,zz,vs_inv);
caxis([ascal1 ascal2])
ylabel('Depth(m)','fontsize',12)
xlabel('X Distance (m)','fontsize',12)
title('S Velocity by WD Inversion','fontsize',12)
axis image
colormap(jet);
h=colorbar;
set(get(h,'Title'),'String','m/s','fontsize',12)
text(t_x,t_y,'(c)','fontsize',12);
set(gca,'fontsize',12)
set(gcf,'color','white');
% print('-dtiff','-r300',['./result/',model_dir,'/P_velocity_iter',num2str(iter),'.tiff']);
saveas(gca,['./result/',model_dir,'/Vs_WD_iter',num2str(iter),'.eps'],'psc2')
print('-dpng','-r300',['./result/',model_dir,'/Vs_vWD_iter',num2str(iter),'.png']);


figure;

subplot(211)
plot(residual./max(residual),'r','LineWidth',2)
caxis([ascal1 ascal2])
ylabel('Misfit','fontsize',12)
xlabel('Iteration (#)','fontsize',12)
title('Data Misfit','fontsize',12)
set(gca,'fontsize',12);
subplot(212)
plot(residual_m./max(residual_m),'LineWidth',2)
caxis([ascal1 ascal2])
ylabel('Misfit','fontsize',12)
xlabel('Iteration (#)','fontsize',12)
title('Model Misfit','fontsize',12)
set(gca,'fontsize',12)
set(gcf,'color','white');
saveas(gca,['./result/',model_dir,'/misfit',num2str(iter),'.eps'],'psc2')
print('-dpng','-r300',['./result/',model_dir,'/misfit',num2str(iter),'.png']);




































