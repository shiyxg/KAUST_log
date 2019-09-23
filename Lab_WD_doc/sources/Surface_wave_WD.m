%% surface wave dispersion curve inversion
%% JIng Li, KAUST, jing.li@kaust.edu.sa
%% CSIM Group

clc;
clear
close all;
addpath('./core/');
model_dir='C_model';
load(['./model/',model_dir,'/C_model.mat']);
%% Choose model, Build the synthetic model
% model_type=82; 
[amp, nz,nx,dx,dt,dtx,vp_d,vp,~,~,vs_d,vs,~,~,den,~,~,~,~,...
    ~,~,~,~,~,rp,rs,rt,~,~,~,~,~,~] = elastic_model(vp,vs);
dz=dx;
%%
%nt=round((nx+nz)*dx/(min(min(vp))/sqrt(3))/dt);  % Define the time steps
nt=1200;
% nt=round((nx+nz/2)*dx/(min(min(vp)))/dt);  % for acoustic test

%% generate source
fr=30;[s,nw]=ricker(fr,dt,nt);
%nbc=round(max(max(vp))/fr/dx*2);  % Add boundary, for fast compute
nbc=90;
%% for mute direct wave
 T_p =  fix(1/dt/(0.5*fr)); 
 T_s = 1.2*T_p; 
%% define acquisition geometry
ds=2; sx=(1:ds:90); sz=zeros(size(sx))+1;[~,ns]=size(sx);
gx=1:2:(nx);   gz=zeros(size(gx))+1;  ng=numel(gx);

%% define wavefield record
dt_wf=dt*5; nt_wf=floor((nt-1)*dt/dt_wf)+1;
pur=-0.05;vsmin=719;vsmax=992;
%T = 6; T1 = 200;   %The paramter for mute P-wave

%%
%seismo_v=zeros(nt,ng,ns);
%deta_seismo_u=zeros(nt,ng,ns);

illum_div=zeros(nz,nx,ns);
cl_img=zeros(nz,nx,ns);
cm_img=zeros(nz,nx,ns);
g_cl=zeros(nz,nx);g_cm=zeros(nz,nx);
g_cl_old=zeros(nz,nx);g_cm_old=zeros(nz,nx);
dk_cl=zeros(nz,nx);dk_cl_old=zeros(nz,nx);
dk_cm=zeros(nz,nx);dk_cm_old=zeros(nz,nx);

parameter_type=0;
g_vp=zeros(nz,nx);g_vs=zeros(nz,nx);
g_vp_old=zeros(nz,nx);g_vs_old=zeros(nz,nx);
dk_vs=zeros(nz,nx);dk_vs_old=zeros(nz,nx);
dk_vp=zeros(nz,nx);dk_vp_old=zeros(nz,nx);

%% Elastic parameters
fd_order=22;fsz=0;source_type='s1';
% free surface
 isfs=1;
 
parallel_init;

%% Set Radon Transform papramter 
 df=0.2;
vmin=400;
vmax=1000;
fmin=10;
fmax=80;
np=601;

%%++++ Calculate the Observed data
parfor is=1:ns
      display(['Synthetic data, shot is=',num2str(is),' ns=',num2str(ns)]);
      [~,seismo_v_d(:,:,is)]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp_d,vs_d,den,isfs,fsz,fd_order,source_type,parameter_type);
% EXtarc the dispersion curve with Radon Transform
 [seismo_v_d(:,:,is)] = csg_sys_mute(seismo_v_d(:,:,is),10,T_s);
[a,b]=size(seismo_v_d(:,:,is));
[ml]=RadonTransfromobs(seismo_v_d(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,b);
[maxpowr,idx]=max(ml(:,:).^13);
cr_0(:,is)=idx;
end

%% Iteration number
iteration=20;
residual=zeros(iteration+1,1);
residual_m=zeros(iteration+1,1);
fre=2*pi*linspace(10,80,351);
%% Define the wavenumber
deta_cc1=zeros(35,45);

%% Define the result file
result_dir0 = '';
result_dir = ['./result/',model_dir,'/',result_dir0];
write_bin([result_dir,'/vs_initial.bin'],vs);
%% ++++++++Start iteration 
%%--------------------------------------------------------------

for k=1:iteration
    tic;
    display(['Elastic_LSM, k=',num2str(k),' iteration=',num2str(iteration)]);
parfor is=1:ns
     display(['Born Modeling, is=',num2str(is),', ns=',num2str(ns)]);
     [~,seismo_v(:,:,is),~]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp,vs,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
     [seismo_v(:,:,is)] = csg_sys_mute(seismo_v(:,:,is),10,T_s);
%% EXtarc the dispersion curve with Radon Transform
    [a,bb]=size(seismo_v(:,:,is));
    [mll(:,:,is),deta_c(:,is),cr_1(:,is)]=RadonTransfrompre(seismo_v(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,bb,cr_0);
%% Cal the model misfit, cr_0 and cr_1 are the  dipersion curve of observed and pred.
    deta_cc(:,is)=fre'.*(-deta_c(:,is))./((cr_1(:,is)+400).*(cr_0(:,is)+400));
 end

 for is=1:ns
 for ii=1:22
     % Select the frequency sample point 
     gg=[1 18 34 51 68 84 101 118 134 151 168 184 201 218 234 251 268 284 301 318 334 351];
     kk=gg(ii);
     deta_cc1(ii+3,is)=deta_cc(kk,is);
 end
 end
  
 %% Calcuate the weight source
for is=1:ns
    ff(:,is)=(fft(s));
for n1=1:35
    ff(n1+1,is)=(ff(n1+1,is)).*(deta_cc1(n1,is));
    ff(1201-n1,is)=(ff(1201-n1,is)).*(deta_cc1(n1,is));
end

    s1(:,is)=ifft(ff(:,is));
end
parfor is=1:ns
    display(['Born Modeling, is=',num2str(is),', ns=',num2str(ns)]);
    [~,~,wavefield_gradient(is)]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s1(:,is),vp,vs,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
end     
    
% Use the penality method to build the objective function
    residual(k)=0.5*sum(deta_c(:).*deta_c(:));

% Define the model misfit
    de=vs-vs_d;
    residual_m(k)=0.5*sum(de(:).*de(:));
   
    display(['residual = ',num2str( residual(k) ),' k=',num2str(k)]);
    res0=residual(k); 
    res0m=residual_m(k);  
 
% Caluate the weight residual data
for is=1:ns
    y(:,is)=linspace(0,30,16); % Define the offset (m)
    deta_seismo_v1(:,:,is)=fft(seismo_v_d(:,round((is)):15+(is),is)); %TRansform shot gather to FK domain
    y1(:,is)=deta_cc1(:,is); % Define the delta k
for ii=1:16
     deta_seismo_v4(:,ii,is)=(((-1i*((y(ii,is)))*exp(-1i*y1(:,is)*((y(ii,is))))/pi/2)));
end
end

for omega=1:30
   deta_seismo_v1(omega+1,:,:)=deta_seismo_v1(omega+1,:,:).*deta_seismo_v4(omega,:,:);
   deta_seismo_v1(1201-omega,:,:)=conj(deta_seismo_v1(omega+1,:,:));
end
   seismo_v_d1=zeros(1200,60,45); % Define the backprogate data
%  seismo_v_d1=seismo_v_d;
for is=1:ns
     deta_seismo_vv(:,:,is)=(ifft(deta_seismo_v1(:,:,is)));
   seismo_v_d1(:,round((is)):15+(is),is)=deta_seismo_vv(:,1:16,is);
end

% Use the RTM calculate the gradient
parfor is=1:ns
    display(['RTM, is=',num2str(is),' ns=',num2str(ns)]);
     [cl_img(:,:,is),cm_img(:,:,is),illum_div(:,:,is)]=e2drtm(wavefield_gradient(is),seismo_v_d1(:,:,is),is,nbc,nt,dtx,dx,dt,gx,gz,s,vp,vs,den,isfs,fsz,fd_order,parameter_type,dt_wf);
end
    g_cl=sum(cl_img,3); g_cm=sum(cm_img,3); g_illum=sum(illum_div,3);
    g_vp=2*vp.*den.*g_cl;g_vs=-4*vs.*den.*g_cl+2*vs.*den.*g_cm;
    [dk_vp,dk_vp_old,dk_vs,dk_vs_old,g_vp_old,g_vs_old]=conjugate_gradiend(dk_vp_old,dk_vs_old,g_vp,g_vp_old,g_vs,g_vs_old,g_illum,k);
    dk_vs=smooth2a(dk_vs,1,2);  % Smooth the Vs gradient
    v_mean=(sum(vs(:).*vs(:)))^0.5;
    g_mean=(sum(dk_vs(:).*dk_vs(:)))^0.5;
    
%     
    alpha=v_mean/g_mean*pur;          
    display(['v_mean=',num2str(v_mean),' g_mean=',num2str(g_mean),' alpha=',num2str(alpha)]);
    
    % Back tracking to find the numerical step length
    f1=0.5;
    vs1=vs+alpha*f1*dk_vs;
    vs1(vs1<vsmin)=vsmin;vs1(vs1>vsmax)=vsmax; 
    vp1=vp;
    
parfor is=1:ns
    [~,seismo_v(:,:,is),~]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp1,vs1,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
    [seismo_v(:,:,is)] = csg_sys_mute(seismo_v(:,:,is),10,T_s);
    [a,bb]=size(seismo_v(:,:,is));
    [~,deta_c(:,is),cr_1(:,is)]=RadonTransfrompre(seismo_v(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,bb,cr_0);  
end

% % Use the penality method to build the objective function
    res1=0.5*sum(deta_c(:).*deta_c(:));
    de=vs1-vs_d;
    res1m=0.5*sum(de(:).*de(:));
   
    display(['f1= ',num2str(f1),' res1= ',num2str(res1)]);
    if res1>res0
        while res1>res0 && f1>0.0001
            f2=f1; res2=res1;
            f1=f1*0.5;
            vs1=vs+alpha*f1*dk_vs;
            vs1(vs1<vsmin)=vsmin;vs1(vs1>vsmax)=vsmax; 
            vp1=vp;

parfor is=1:ns
       [~,seismo_v(:,:,is),~]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp1,vs1,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
       [a,bb]=size(seismo_v(:,:,is));
        [seismo_v(:,:,is)] = csg_sys_mute(seismo_v(:,:,is),10,T_s);
       [~,deta_c(:,is),cr_1(:,is)]=RadonTransfrompre(seismo_v(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,bb,cr_0);  
end

% Use the penality method to build the objective function
    res1=0.5*sum(deta_c(:).*deta_c(:));
    de=vs1-vs_d;
    res1m=0.5*sum(de(:).*de(:));
     display(['f1= ',num2str(f1),' res1= ',num2str(res1)]);
        end
    else
        f2=f1*2;
        vs1=vs+alpha*f2*dk_vs; 
        vs1(vs1<vsmin)=vsmin;vs1(vs1>vsmax)=vsmax; 
        vp1=vp;

parfor is=1:ns
       [~,seismo_v(:,:,is),~]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp1,vs1,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
        [seismo_v(:,:,is)] = csg_sys_mute(seismo_v(:,:,is),10,T_s);
       [a,bb]=size(seismo_v(:,:,is));
       [~,deta_c(:,is),cr_1(:,is)]=RadonTransfrompre(seismo_v(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,bb,cr_0);  
end

% Use the penality method to build the objective function

    res2=0.5*sum(deta_c(:).*deta_c(:));
    de=vs1-vs_d;
    res2m=0.5*sum(de(:).*de(:));

        display(['f2= ',num2str(f2),' res2= ',num2str(res2)]);
end
    gama=(f1^2*(res0-res2)+f2^2*(res1-res0))/(2*res0*(f1-f2)+2*res1*f2-2*res2*f1);
    display(['gama= ',num2str(gama),' numerical step_length= ',num2str(gama*alpha)]);

    vs1=vs+alpha*gama*dk_vs;

    vs1(vs1<vsmin)=vsmin;vs1(vs1>vsmax)=vsmax; 
    vp1=vp;

parfor is=1:ns
       [~,seismo_v(:,:,is),~]=staggerfd(is,nbc,nt,dtx,dx,dt,sx(is),sz(is),gx,gz,s,vp1,vs1,den,isfs,fsz,fd_order,source_type,parameter_type,dt_wf,nt_wf);
        [seismo_v(:,:,is)] = csg_sys_mute(seismo_v(:,:,is),10,T_s);
        [a,bb]=size(seismo_v(:,:,is));
        [mll(:,:,is),deta_c(:,is),cr_1(:,is)]=RadonTransfrompre(seismo_v(:,:,is),is,df,dt,np,vmin,vmax,fmin,fmax,a,bb,cr_0);  
end

% Use the penality method to build the objective function
     res3=0.5*sum(deta_c(:).*deta_c(:));
     de=vs1-vs_d;
    res3m=0.5*sum(de(:).*de(:));

    display(['res3= ',num2str(res3)]);
    if (res3>res1 || res3>res2)
        if res1>res2
            res0=res2;
            res0m=res2m;
            %lamta=f2;
            gama=f2; 
        else
            res0=res1;
             res0m=res1m;
            %lamta=f1;
            gama=f1; 
        end
        vs1=vs+alpha*gama*dk_vs;      
     vs1(vs1<vsmin)=vsmin;vs1(vs1>vsmax)=vsmax; 
      vp1=vp;
    else
        res0=res3;
        res0m=res3m;
    end
    vp=vp1;
    vs=vs1;
    toc;
    write_bin([result_dir,'/Vs_inv_',num2str(k),'.bin'],vs);
     write_bin([result_dir,'/Dis_',num2str(k),'.bin'],cr_1);
end
residual(iteration+1)=res0;
residual_m(iteration+1)=res0m;
 write_bin([result_dir,'/data_misfit.bin'],residual);
 write_bin([result_dir,'/model_misfit.bin'],residual_m);
parallel_stop;

