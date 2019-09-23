%
 function [ml]=RadonTransfromobs(seismo_v1,is,df,dt,np,vmin,vmax,fmin,fmax,a,b)
%#######################################################################!

  [seismo_v1] = fk_filter(seismo_v1, dt, 2);
   uxt=seismo_v1(:,round((is)):15+(is));
   x=linspace(0,30,16);
[poinum,tranum]=size(uxt);
di=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [uxt,tranum,poinum,dt,x,fmin,fmax,df,vmin,vmax,np,outname]=luo_taup_1981(uxt,tranum,poinum,dt,x,fmin,fmax,df,vmin,vmax,np,outname);


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%                                                                                !
%                                �ӳ���subroutine                                !
%                                                                                !
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

i=complex(0,1);
pmin=0.0000;
pmax=0.01;
pmax1=1.0./vmin;
if(pmax1>pmax)
pmax=pmax1;
end;

%************************parfor is=1:30*************************************************%
% zero-padding
 uxt=uxt';
m=(tranum);
nn0=(poinum);
ccn=fix(((1./df./dt-1-nn0))+1);
zeros1=zeros(m,ccn);
uxt_m=(size(uxt,1));
uxt_n=(size(zeros1,2)+size(uxt,2));
uxt_uxt=zeros(uxt_m,uxt_n);

for k=1:m;
for t=1:uxt_n;
if(t<=a)
uxt_uxt(k,t)=uxt(k,t);
end
if(t>a)
uxt_uxt(k,t)=zeros1(k,t-a);
end
end
end

m=fix(size(uxt_uxt,1));
n=fix(size(uxt_uxt,2));
%single trace fourier transform
d=zeros(m,n);
abs_d=zeros(m,n);
for luoi=1:m;
% ���и���Ҷ�任(������) D(luoi,:)=fft(uxt_uxt(luoi,:),n)
d(luoi,:)=fft(uxt_uxt(luoi,:));
end

%initialize parameters
lf=round(fmin./df)+1;
nf=round(fmax./df)+1;
%  print*,lf,nf
%m�ǵ���
nx=fix(m);

dp=(pmax-pmin)./(np-1);
%p(1,451),x(1,24)
p(1,:)=(pmin+([1:np]-1).*dp);
ll0_n=fix(size(p,2));
ll0_m=fix(size(x,2));
%  print*,ll0_n,ll0_m
ll0=zeros(ll0_n,ll0_m);
pp=p';
%print*,pp
%ll0(451,24)  �˴��൱��������(������ԭʼ��taup�任����)
ll0=i.*2.*pi.*df.*(pp*x);
%print*,ll0
%mm(451,5000)
mm=zeros(np,n+1);
abs_mm=zeros(np,n+1);
l=zeros(np,nx);
mm(:)=0;
%pause
%*************************************************************************%
% no sparseness  constraint
%  [nt,nh] = size(d);
%   dq = (qmax-qmin)/(nq-1);
%   q = qmin+dq*[0:1:nq-1];
%  nq = max(size(q));
% nq=50;parfor is=1:30
% mu=0.1;
%  Q = eye(nq)*m;
%lf=26,nf=251
for luoj=lf:nf;
l([1:np],[1:nx])=exp(ll0.*(luoj-1));
%print*,L
%L*D(1:nx,luoj)
mm([1:np],luoj)=(l*d([1:nx],luoj));
% mm([1:np],luoj)=(l*d([1:nx],luoj));
% A=l*l+mu*Q;
% mm([1:np],luoj)=A\mm([1:np],luoj);
end
% abs_mm=abs(mm)
% open(unit=120,file='mm.txt')
% write(120,*)abs_mm
% close(120)
%pause
%************************************************************************%
% translate to frequency-velocity domain
p1=zeros(1,np);
p1(1,1)=1000000000;
p1(1,[2:np])=1.0./p(1,[2:np]);
p2_size=fix(fix(fix(vmax-vmin)./1)+1);
p2=zeros(1,p2_size);
for j=vmin:vmax;
p2(j-399)=j;
end; j=vmax+1;
ndata=fix(p2_size);
%p2=vmin:1:vmax
ml=zeros(np,fix(nf));
%pause    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for luoi=lf:nf;
% ��Ƶɢ���߽��в�ֵml(451,251) (NDATA,XDATA,FDATA,N,XVEC,VALUE)
ml(:,luoi)=interp1(p1,abs(mm(:,luoi)),p2,'spline');
% [ndata,p1,dumvar3,p2_size,p2,ml(:,luoi)]=csiez(ndata,p1,abs(mm(:,luoi)),p2_size,p2,ml(:,luoi));
% tempmaxval=ml(:,luoi);
% maxv=max(tempmaxval(:));
% % ����ݽ��й�һ������lf:nf
 ml(:,luoi)=ml(:,luoi)./max(ml(:,luoi));
end
 ml=abs(ml(:,51:401));
 
%   for i=1:351
%      cr(i)=ml(round(cr_01(i)),i);
%  end
 
 
%return
