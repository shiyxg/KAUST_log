function [amp, nz,nx,dx,dt,dtx,vp,vp_ss,vp_homo,refl_vp,vs,vs_ss,vs_homo,refl_vs,den,ca,cm,cm_ss,cm_homo,refl_cm,cl,cl_ss,cl_homo,refl_cl,rp,rs,rt,rp_ss,rs_ss,rt_ss,rp_homo,rs_homo,rt_homo] = elastic_model(vp,vs)
% By Zongcai Feng  2015.4.28

% if model_type==82
    amp=0.5;
%     load C_model.mat
%     vp=vp(8:32,11:180);
%     vs=vs(8:32,11:180);
    nx=240*amp;dx=1.0;nz=52*amp;dt=0.00025;dtx=dt/dx;
    x=(0:nx-1)*dx;z=(0:nz-1)*dx;
%     vp=ones(nz,nx)*1200;vs=ones(nz,nx)*700;
    den=ones(nz,nx);
    
   %vp_ss=vp; vs_ss=vs;
%     nt=1000;
% generate source
   fr=30;
    vp=floor(vp);
    vs=floor(vs);
  
  vp_ss=vp;
%     [vs_ss,~]=vel_smooth(vs,3,5,150);
  for i=1:120
      vs_ss(:,i)=linspace(719,992,26);
  end

 Q_rev=0.00001*ones(nz,nx);
 Q_revs=0.00001*ones(nz,nx);
 [Q_rev]=smooth2a(Q_rev,2,2);
 [Q_revs]=smooth2a(Q_revs,2,2);
%      Q_rev(10:13,1:30)=0.03;
%      Q_revs(10:13,1:30)=0.03;
%      Q_rev(20:30,:)=0.05;
%      Q_revs(20:30,:)=0.05;
%  Q_rev(round(nz*1.8/4):round(nz*1.8/4)+1,:)=0.05;
%  Q_revs(round(nz*3/4):round(nz*3/4)+1,:)=0.05;
 
rt=(sqrt(1+Q_rev.^2)-Q_rev)/(2*pi*fr);
rp=1/(2*pi*fr)^2./rt;
%rs=1/(2*pi*fr)^2./rt;
rs=(1+2*pi*fr*(1./Q_revs).*rt)./(2*pi*fr.*(1./Q_revs)-rt.*(2*pi*fr)^2);

%    [rp_ss,~]=vel_smooth(rp,1,1,1);
%   [rs_ss,~]=vel_smooth(rs,1,1,1);  
%     [rt_ss,~]=vel_smooth(rt,1,1,1); 
 rt_ss=rt; rp_ss=rp;rs_ss=rs;
    
    refl_vp=vp-vp_ss;  refl_vs=vs-vs_ss;
    
    [ca,cm,cl]=calparam(vp,vs,den);
    [~,cm_ss,cl_ss]=calparam(vp_ss,vs_ss,den);
    
    refl_cm=cm-cm_ss; refl_cl=cl-cl_ss;
    
    cm_homo=zeros(size(cm))+min(cm(:));    cl_homo=zeros(size(cl))+min(cl(:));
    vp_homo=zeros(size(vp))+min(vp(:));    vs_homo=zeros(size(vs))+min(vs(:));
    rt_homo=zeros(size(rt))+max(rt(:));
    rp_homo=zeros(size(rp))+min(rp(:));
    rs_homo=zeros(size(rs))+min(rs(:));
    
    %     figure(1);
    %     subplot(221);imagesc(x,z,vp);colorbar; xlabel('X (m)','fontsize',14); ylabel('Z (m)','fontsize',14);title('a) True P-wave velocity','fontsize',14);
    %     subplot(223);imagesc(x,z,vp_ss);colorbar;xlabel('X (m)','fontsize',14); ylabel('Z (m)','fontsize',14);title('b) Migration P-wave velocity','fontsize',14);
    %     subplot(222);imagesc(x,z,vs);colorbar; xlabel('X (m)','fontsize',14); ylabel('Z (m)','fontsize',14);title('c) S-wave Velocity','fontsize',14);
    %     subplot(224);imagesc(x,z,vs_ss);colorbar; xlabel('X (m)','fontsize',14); ylabel('Z (m)','fontsize',14);title('d) Migration S-wave velocity','fontsize',14);
    %
    figure(1);
    subplot(221);imagesc(x,z,vp);colorbar;  ylabel('Z (m)','fontsize',14);title('a) True P-wave velocity','fontsize',14);
    subplot(223);imagesc(x,z,vp_ss);colorbar;xlabel('X (m)','fontsize',14); ylabel('Z (m)','fontsize',14);title('b) Start P-wave velocity','fontsize',14);
    subplot(222);imagesc(x,z,vs);colorbar;title('c) S-wave Velocity','fontsize',14);
    subplot(224);imagesc(x,z,vs_ss);colorbar; xlabel('X (m)','fontsize',14); title('d) Start S-wave velocity','fontsize',14);
end
% end



