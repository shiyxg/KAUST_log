% 2D elastic modeling by staggered grid method
% @version 2 2014-09-26
% @author Bowen Guo

%Born modeling by Zongcai Feng 2015-03-24
%add parameter: vp_refl,vs_refl

function [cl_img,cm_img,illum_div]=e2drtm(wavefield_gradient,seismo_w,is,nbc,nt,dtx,dx,dt,gx,gz,s,vp,vs,den,isfs,fsz,fd_order,parameter_type,dt_wf)



% INPUT
% is:shot number
% sx,sz:shot coordinate
% gx,gx:geophone coordinate
% nbc: padding number (used in pad and ABC)
% nt: time length
% dtx: dt/dx,dt is the time interval, dx is the space interval
% s:source wavelet
% vp: p wave velocity
% vs: s wave velocity
% vp_refl: P wave reflectivity, vp=vp0(1+vp_refl)
% vs_refl: s wave reflectivity, vs=vs0(1+vs_refl)
% den: density
% isfs: 0:no free surface; 1: free surface
% fsz: free surface layer
% fd_order:finite difference accuracy order (24,26,28)
% source_type: different ways to add sources:p/s/w/tau_zz
% p: Initiate Strong P wave, weak S wave
% tau_zz/w: Both P and S wave
% s: Strong S wave, weak P wave
% When using free surface boundary conditio, source_type=p is
% recommended.

% OUTPUT
% seismo_u: horizontal displacement component recorded data
% seismo_w: vertical displacement component recorded data


% Calculate lambda, mu based on density, p/s wave veolcity
% ca: lambda+2*mu; cl:lambda; cm: mu
%[ca,cm,cl]=calparam(vp,vs,den);
if (nargin)<=18
    dt_wf=dt;
end

if parameter_type==0
    [ca,cm,cl]=calparam(vp,vs,den);
end
if parameter_type==1
    ca=cl+2*cm;
end

cl_img=zeros(size(vp)); cm_img=zeros(size(vp)); illum_div=zeros(size(vp));%illum_curl=zeros(size(vp));

% staggered grid finite difference coeffcients
S21=1.0;
S41=9.0/8.0;S42=-1.0/24.0;
S61=1.17187;S62=-6.51042e-2;S63=4.68750e-3;
S81=1.19629;S82=-7.97526e-2;S83=9.57031e-3;S84=-6.97545e-4;

tic;

ng=numel(gx);
% pad means to expand the model space in order to add the absorbing
% boundary condition
if (isfs)
    pad_top=(fd_order-20)/2+1;
else
    pad_top=nbc;
end

cm=pad(cm,nbc,isfs,pad_top);
cl=pad(cl,nbc,isfs,pad_top);
ca=pad(ca,nbc,isfs,pad_top);
den=pad(den,nbc,isfs,pad_top);


%Adjust source and receiver position because of free surface
if (isfs)
%     if (sz==fsz)
%         sz=sz+1;
%     end
    if (gz==fsz)
        gz=gz+1;
    end
end

% change source/geophone position because of pad
gx=gx+nbc;gz=gz+pad_top;

if (isfs)
    fsz=fsz+pad_top;
end

[nzbc,nxbc]=size(cm);

%*_p means purturb wavefeld
uu_b=zeros(nzbc,nxbc); % horizontal displacement wavefield
ww_b=zeros(nzbc,nxbc); % vertical displacement wavefield
xx_b=zeros(nzbc,nxbc); % tau_xx wavefield
zz_b=zeros(nzbc,nxbc); % tau_zz wavefield
xz_b=zeros(nzbc,nxbc); % tau_xz wavefield

% calculate dt/dx/dens
b=dtx./den;
% damp is used in the absorbing boundary condition
vmin=min(min(vp(:),vs(:)));
damp=damp_circle(vmin,nzbc,nxbc,nbc,dx,isfs,pad_top);
temp=1-damp*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Such scheme is abandoned.
% Adjust material parameter according tosave WD_test_iluu.mat vs_d vs vs1 cr_1 cr_0 residual residual_m dk_vs vp_d
% 'Free-surface boundary conditions for elastic staggered-grid modeling
% schemes' by Rune Mittet
if (isfs)
   cm(fsz,:)=0.5*cm(fsz,:);
   ca(fsz,:)=2*cm(fsz,:);
   cl(fsz,:)=0.0;
   b(fsz,:)=2*b(fsz,:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (fd_order==22)
    k=(2:nzbc-1);i=(2:nxbc-1);
elseif (fd_order==24)
    k=(3:nzbc-2);i=(3:nxbc-2);
elseif (fd_order==26)
    k=(4:nzbc-3);i=(4:nxbc-3);
elseif (fd_order==28)
    k=(5:nzbc-4);i=(5:nxbc-4);
end

%Back propogate modeling  by Zongcai
for it=nt-1:-1:1
    %   display(['RTM it=',num2str(it)]);
    % Calculate particle velocity
    if (fd_order==22)
        uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*( S21*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+...
            S21*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S21*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i)) );
        
        ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*( S21*(cl(k+1,i).*xx_b(k+1,i)-cl(k,i).*xx_b(k,i))+...
            +S21*(ca(k+1,i).*zz_b(k+1,i)-ca(k,i).*zz_b(k,i))+S21*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i).*xz_b(k,i)) );
    elseif (fd_order==24)
        uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S41*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1)) + S42*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2)))+...
            (S41*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S42*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2)))+...
            (S41*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i))+S42*(cm(k+1,i).*xz_b(k+1,i)-cm(k-2,i).*xz_b(k-2,i))));
        
        ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S41*(cl(k+1,i).*xx_b(k+1,i)-cl(k,i).*xx_b(k,i))+S42*(cl(k+2,i).*xx_b(k+2,i)-cl(k-1,i).*xx_b(k-1,i)))+...
            (S41*(ca(k+1,i).*zz_b(k+1,i)-ca(k,i).*zz_b(k,i))+S42*(ca(k+2,i).*zz_b(k+2,i)-ca(k-1,i).*zz_b(k-1,i)))+...
            (S41*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i).*xz_b(k,i))+S42*(cm(k,i+2).*xz_b(k,i+2)-cm(k,i-1).*xz_b(k,i-1))));
    elseif (fd_order==26)
        uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S61*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+S62*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2))+...
            +S63*(ca(k,i+2).*xx_b(k,i+2)-ca(k,i-3).*xx_b(k,i-3)))+...
            (S61*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S62*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2))+S63*(cl(k,i+2).*zz_b(k,i+2)-cl(k,i-3).*zz_b(k,i-3)))+...
            (S61*( cm(k,i).*xz_b(k,i)- cm(k-1,i).*xz_b(k-1,i))+S62*( cm(k+1,i).*xz_b(k+1,i)- cm(k-2,i).*xz_b(k-2,i))+S63*( cm(k+2,i).*xz_b(k+2,i)- cm(k-3,i).*xz_b(k-3,i))));
        
        ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S61*(cl(k+1,i).*xx_b(k+1,i)-cl(k,i).*xx_b(k,i))+S62*(cl(k+2,i).*xx_b(k+2,i)-cl(k-1,i).*xx_b(k-1,i))+...
            +S63*(cl(k+3,i).*xx_b(k+3,i)-cl(k-2,i).*xx_b(k-2,i)))+...
            (S61*(ca(k+1,i).*zz_b(k+1,i)-ca(k,i).*zz_b(k,i))+S62*(ca(k+2,i).*zz_b(k+2,i)-ca(k-1,i).*zz_b(k-1,i))+S63*(ca(k+3,i).*zz_b(k+3,i)-ca(k-2,i).*zz_b(k-2,i)))+...
            (S61*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i).*xz_b(k,i))+S62*(cm(k,i+2).*xz_b(k,i+2)-cm(k,i-1).*xz_b(k,i-1))+S63*(cm(k,i+3).*xz_b(k,i+3)-cm(k,i-2).*xz_b(k,i-2))));
    elseif (fd_order==28)
        uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S81*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+S82*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2))+...
            +S83*(ca(k,i+2).*xx_b(k,i+2)-ca(k,i-3).*xx_b(k,i-3))+S84*(ca(k,i+3).*xx_b(k,i+3)-ca(k,i-4).*xx_b(k,i-4)))+...
            (S81*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S82*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2))+S83*(cl(k,i+2).*zz_b(k,i+2)-cl(k,i-3).*zz_b(k,i-3))+...
            S84*(cl(k,i+3).*zz_b(k,i+3)-cl(k,i-4).*zz_b(k,i-4)))+...
            (S81*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i))+S82*(cm(k+1,i).*xz_b(k+1,i)-cm(k-2,i).*xz_b(k-2,i))+S83*(cm(k+2,i).*xz_b(k+2,i)-cm(k-3,i).*xz_b(k-3,i))+...
            S84*(cm(k+3,i).*xz_b(k+3,i)-cm(k-4,i).*xz_b(k-4,i))));
        
        ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S81*(cl(k+1,i).*xx_b(k+1,i)-cl(k,i).*xx_b(k,i))+S82*(cl(k+2,i).*xx_b(k+2,i)-cl(k-1,i).*xx_b(k-1,i))+...
            +S83*(cl(k+3,i).*xx_b(k+3,i)-cl(k-2,i).*xx_b(k-2,i))+S84*(cl(k+4,i).*xx_b(k+4,i)-cl(k-3,i).*xx_b(k-3,i)))+...
            (S81*(ca(k+1,i).*zz_b(k+1,i)-ca(k,i).*zz_b(k,i))+S82*(ca(k+2,i).*zz_b(k+2,i)-ca(k-1,i).*zz_b(k-1,i))+S83*(ca(k+3,i).*zz_b(k+3,i)-ca(k-2,i).*zz_b(k-2,i))+...
            S84*(ca(k+4,i).*zz_b(k+4,i)-ca(k-3,i).*zz_b(k-3,i)))+...
            (S81*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i).*xz_b(k,i))+S82*(cm(k,i+2).*xz_b(k,i+2)-cm(k,i-1).*xz_b(k,i-1))+S83*(cm(k,i+3).*xz_b(k,i+3)-cm(k,i-2).*xz_b(k,i-2))+...
            S84*(cm(k,i+4).*xz_b(k,i+4)-cm(k,i-3).*xz_b(k,i-3))));
    end
    
    
    % Free surface boundary condition for velocity
%         if (isfs)
%         ww_b(fsz-1,i)=ww_b(fsz,i)+cl(fsz,i)./ca(fsz,i).*(uu_b(fsz,i+1)-uu_b(fsz,i));
%         %uu_b(fsz-1,i)=uu_b(fsz,i)+(uu_b(fsz+1,i)-uu_b(fsz,i)+ww_b(fsz,i+1)-ww_b(fsz,i))+(ww_b(fsz-1,i+1)-ww_b(fsz-1,i));
%         end
    
    for ig=1:ng
%          uu_b(gz(ig),gx(ig))=uu_b(gz(ig),gx(ig))+dt*seismo_u(it,ig)/den(gz(ig),gx(ig));
        ww_b(gz(ig),gx(ig))=ww_b(gz(ig),gx(ig))+dt*seismo_w(it,ig)/den(gz(ig),gx(ig));
    end
    
    % update stress
    if  (fd_order==22)
        xx_b(k,i)=temp(k,i).*xx_b(k,i)-dtx*S21*(uu_b(k,i+1)-uu_b(k,i));
        zz_b(k,i)=temp(k,i).*zz_b(k,i)-dtx*S21*(ww_b(k,i)-ww_b(k-1,i));
        xz_b(k,i)=temp(k,i).*xz_b(k,i)-dtx*S21*(uu_b(k+1,i)-uu_b(k,i))-dtx*S21*(ww_b(k,i)-ww_b(k,i-1));
    elseif (fd_order==24)
        xx_b(k,i)=temp(k,i).*xx_b(k,i)-dtx*(S41*(uu_b(k,i+1)-uu_b(k,i))+S42*(uu_b(k,i+2)-uu_b(k,i-1)));
        zz_b(k,i)=temp(k,i).*zz_b(k,i)-dtx*(S41*(ww_b(k,i)-ww_b(k-1,i))+S42*(ww_b(k+1,i)-ww_b(k-2,i)));
        xz_b(k,i)=temp(k,i).*xz_b(k,i)-...
            dtx*(S41*(uu_b(k+1,i)-uu_b(k,i))+S42*(uu_b(k+2,i)-uu_b(k-1,i)))-...
            dtx*(S41*(ww_b(k,i)-ww_b(k,i-1))+S42*(ww_b(k,i+1)-ww_b(k,i-2)));
    elseif (fd_order==26)
        xx_b(k,i)=temp(k,i).*xx_b(k,i)-dtx*(S61*(uu_b(k,i+1)-uu_b(k,i))+S62*(uu_b(k,i+2)-uu_b(k,i-1))+S63*(uu_b(k,i+3)-uu_b(k,i-2)));
        zz_b(k,i)=temp(k,i).*zz_b(k,i)-dtx*(S61*(ww_b(k,i)-ww_b(k-1,i))+S62*(ww_b(k+1,i)-ww_b(k-2,i))+S63*(ww_b(k+2,i)-ww_b(k-3,i)));
        xz_b(k,i)=temp(k,i).*xz_b(k,i)-...
            dtx*(S61*(uu_b(k+1,i)-uu_b(k,i))+S62*(uu_b(k+2,i)-uu_b(k-1,i))+S63*(uu_b(k+3,i)-uu_b(k-2,i)))-...
            dtx*(S61*(ww_b(k,i)-ww_b(k,i-1))+S62*(ww_b(k,i+1)-ww_b(k,i-2))+S63*(ww_b(k,i+2)-ww_b(k,i-3)));
    elseif (fd_order==28)
        xx_b(k,i)=temp(k,i).*xx_b(k,i)-dtx*(S81*(uu_b(k,i+1)-uu_b(k,i))+S82*(uu_b(k,i+2)-uu_b(k,i-1))+...
            S83*(uu_b(k,i+3)-uu_b(k,i-2))+S84*(uu_b(k,i+4)-uu_b(k,i-3)));
        zz_b(k,i)=temp(k,i).*zz_b(k,i)-dtx*(S81*(ww_b(k,i)-ww_b(k-1,i))+S82*(ww_b(k+1,i)-ww_b(k-2,i))+...
            S83*(ww_b(k+2,i)-ww_b(k-3,i))+S84*(ww_b(k+3,i)-ww_b(k-4,i)));
        xz_b(k,i)=temp(k,i).*xz_b(k,i)-...
            dtx*(S81*(uu_b(k+1,i)-uu_b(k,i))+S82*(uu_b(k+2,i)-uu_b(k-1,i))+S83*(uu_b(k+3,i)-uu_b(k-2,i))+S84*(uu_b(k+4,i)-uu_b(k-3,i)))-...
            dtx*(S81*(ww_b(k,i)-ww_b(k,i-1))+S82*(ww_b(k,i+1)-ww_b(k,i-2))+S83*(ww_b(k,i+2)-ww_b(k,i-3))+S84*(ww_b(k,i+3)-ww_b(k,i-4)));
        
    end
    
    % free surface boundary condition
    if (isfs)
        zz_b(fsz,:)=0.0;
%         xz_b(fsz,:)=0.0;
%   
%           xz_b(fsz-1,:)=-xz_b(fsz+1,:);
    end

    %image condition
    in_wf=dt_wf/dt;
    if mod(it-1,in_wf)==0
        it_record=1+floor((it-1)/in_wf);
        fux=wavefield_gradient.fux(:,:,it_record);
        fuz=wavefield_gradient.fuz(:,:,it_record);
        bwx=wavefield_gradient.bwx(:,:,it_record);
        bwz=wavefield_gradient.bwz(:,:,it_record);
%         fuu=wavefield_gradient.uu(:,:,it_record);
%         bww=wavefield_gradient.ww(:,:,it_record);
        [cl_img,cm_img]=image_condition(cl_img,cm_img,xx_b,zz_b,xz_b,fux,fuz,bwx,bwz,pad_top,nbc,dt);
        
%        fuu=fuu(pad_top+1:end-nbc,nbc+1:end-nbc);
%        bww=bww(pad_top+1:end-nbc,nbc+1:end-nbc);
       
     % illum_div=illum_div+(fuu.^2+bww.^2);
        %illum_div=illum_div+(fux.^2+bwz.^2);
        illum_div=illum_div+(fux+bwz).^2;
        % illum_curl=illum_curl+(fuz-bwx).^2;
    end
    
        % %    plot the snapshot of the wavefield
    %            if it/10 == round(it/10);
    %                 plotit(dx,ww_b,nxbc,nzbc,it,dt);
    %             end
    
end
toc
%display(num2str(is),'th shot');
end

%     if (fd_order==22)
%         uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*( S21*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+...
%             S21*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S21*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i)) );
%
%         ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*( S21*(cl(k,i).*xx_b(k,i)-cl(k-1,i).*xx_b(k-1,i))+...
%             +S21*(ca(k,i).*zz_b(k,i)-ca(k-1,i).*zz_b(k-1,i))+S21*(cm(k,i).*xz_b(k,i)-cm(k,i-1).*xz_b(k,i-1)) );
%     elseif (fd_order==24)
%         uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S41*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1)) + S42*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2)))+...
%             (S41*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S42*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2)))+...
%             (S41*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i))+S42*(cm(k+1,i).*xz_b(k+1,i)-cm(k-2,i).*xz_b(k-2,i))));
%
%         ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S41*(cl(k,i).*xx_b(k,i)-cl(k-1,i).*xx_b(k-1,i))+S42*(cl(k+1,i).*xx_b(k+1,i)-cl(k-2,i).*xx_b(k-2,i)))+...
%             (S41*(ca(k,i).*zz_b(k,i)-ca(k-1,i).*zz_b(k-1,i))+S42*(ca(k+1,i).*zz_b(k+1,i)-ca(k-2,i).*zz_b(k-2,i)))+...
%             (S41*(cm(k,i).*xz_b(k,i)-cm(k,i-1).*xz_b(k,i-1))+S42*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i-2).*xz_b(k,i-2))));
%     elseif (fd_order==26)
%         uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S61*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+S62*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2))+...
%             +S63*(ca(k,i+2).*xx_b(k,i+2)-ca(k,i-3).*xx_b(k,i-3)))+...
%             (S61*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S62*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2))+S63*(cl(k,i+2).*zz_b(k,i+2)-cl(k,i-3).*zz_b(k,i-3)))+...
%             (S61*( cm(k,i).*xz_b(k,i)- cm(k-1,i).*xz_b(k-1,i))+S62*( cm(k+1,i).*xz_b(k+1,i)- cm(k-2,i).*xz_b(k-2,i))+S63*( cm(k+2,i).*xz_b(k+2,i)- cm(k-3,i).*xz_b(k-3,i))));
%
%         ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S61*(cl(k,i).*xx_b(k,i)-cl(k-1,i).*xx_b(k-1,i))+S62*(cl(k+1,i).*xx_b(k+1,i)-cl(k-2,i).*xx_b(k-2,i))+...
%             +S63*(cl(k+2,i).*xx_b(k+2,i)-cl(k-3,i).*xx_b(k-3,i)))+...
%             (S61*(ca(k,i).*zz_b(k,i)-ca(k-1,i).*zz_b(k-1,i))+S62*(ca(k+1,i).*zz_b(k+1,i)-ca(k-2,i).*zz_b(k-2,i))+S63*(ca(k+2,i).*zz_b(k+2,i)-ca(k-3,i).*zz_b(k-3,i)))+...
%             (S61*(cm(k,i).*xz_b(k,i)-cm(k,i-1).*xz_b(k,i-1))+S62*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i-2).*xz_b(k,i-2))+S63*(cm(k,i+2).*xz_b(k,i+2)-cm(k,i-3).*xz_b(k,i-3))));
%     elseif (fd_order==28)
%         uu_b(k,i)=temp(k,i).*uu_b(k,i)-b(k,i).*((S81*(ca(k,i).*xx_b(k,i)-ca(k,i-1).*xx_b(k,i-1))+S82*(ca(k,i+1).*xx_b(k,i+1)-ca(k,i-2).*xx_b(k,i-2))+...
%             +S83*(ca(k,i+2).*xx_b(k,i+2)-ca(k,i-3).*xx_b(k,i-3))+S84*(ca(k,i+3).*xx_b(k,i+3)-ca(k,i-4).*xx_b(k,i-4)))+...
%             (S81*(cl(k,i).*zz_b(k,i)-cl(k,i-1).*zz_b(k,i-1))+S82*(cl(k,i+1).*zz_b(k,i+1)-cl(k,i-2).*zz_b(k,i-2))+S83*(cl(k,i+2).*zz_b(k,i+2)-cl(k,i-3).*zz_b(k,i-3))+...
%             S84*(cl(k,i+3).*zz_b(k,i+3)-cl(k,i-4).*zz_b(k,i-4)))+...
%             (S81*(cm(k,i).*xz_b(k,i)-cm(k-1,i).*xz_b(k-1,i))+S82*(cm(k+1,i).*xz_b(k+1,i)-cm(k-2,i).*xz_b(k-2,i))+S83*(cm(k+2,i).*xz_b(k+2,i)-cm(k-3,i).*xz_b(k-3,i))+...
%             S84*(cm(k+3,i).*xz_b(k+3,i)-cm(k-4,i).*xz_b(k-4,i))));
%
%         ww_b(k,i)=temp(k,i).*ww_b(k,i)-b(k,i).*((S81*(cl(k,i).*xx_b(k,i)-cl(k-1,i).*xx_b(k-1,i))+S82*(cl(k+1,i).*xx_b(k+1,i)-cl(k-2,i).*xx_b(k-2,i))+...
%             +S83*(cl(k+2,i).*xx_b(k+2,i)-cl(k-3,i).*xx_b(k-3,i))+S84*(cl(k+3,i).*xx_b(k+3,i)-cl(k-4,i).*xx_b(k-4,i)))+...
%             (S81*(ca(k,i).*zz_b(k,i)-ca(k-1,i).*zz_b(k-1,i))+S82*(ca(k+1,i).*zz_b(k+1,i)-ca(k-2,i).*zz_b(k-2,i))+S83*(ca(k+2,i).*zz_b(k+2,i)-ca(k-3,i).*zz_b(k-3,i))+...
%             S84*(ca(k+3,i).*zz_b(k+3,i)-ca(k-4,i).*zz_b(k-4,i)))+...
%             (S81*(cm(k,i).*xz_b(k,i)-cm(k,i-1).*xz_b(k,i-1))+S82*(cm(k,i+1).*xz_b(k,i+1)-cm(k,i-2).*xz_b(k,i-2))+S83*(cm(k,i+2).*xz_b(k,i+2)-cm(k,i-3).*xz_b(k,i-3))+...
%             S84*(cm(k,i+3).*xz_b(k,i+3)-cm(k,i-4).*xz_b(k,i-4))));
%     end
