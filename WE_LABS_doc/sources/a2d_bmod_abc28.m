function [seis,bc_top,bc_bottom,bc_left,bc_right,bc_p_nt,bc_p_nt_1]...
    =a2d_bmod_abc28(v,refl,nbc,dx,nt,dt,s,sx,sz,gx,gz,isFS)
%  Copyright (C) 2010 Center for Subsurface Imaging and Fluid Modeling (CSIM),
%  King Abdullah University of Science and Technology, All rights reserved.
%
%  author:   Xin Wang
%  email:    xin.wang@kaust.edu.sa
%  date:     Sep 26, 2012
%  purpose:  2DTDFD born solution to acoustic wave equation with accuracy of 2-8
%            use the absorbing boundary condition
%
%  IN   v(:,:) -- velocity,      refl(:,:)   -- reflectivity 
%       nbc    -- grid number of boundary
%       dx     -- grid intervel, nt          -- number of sample
%       dt     -- time interval, s(:)        -- wavelet
%       sx,sz  -- src position,  gx(:),gz(:) -- rec position
%       isFS   -- if is Free Surface condition
%  OUT  seis(:,:)          -- Output born modeling seismogram
%       bc_top(:,:,:)      -- array for storing top boundary condition
%       bc_bottom(:,:,:)   -- array for storing bottom boundary condition
%       bc_left(:,:,:)     -- array for storing left boundary condition
%       bc_right(:,:,:)    -- array for storing right boundary condition
%       bc_p_nt(:,:)       -- array for storing the last time step wavefield
%       bc_p_nt_1(:,:)     -- array for storing the second last time step wavefield

seis=zeros(nt,numel(gx));[nz,nx]=size(v);
ng=numel(gx);
c1 = -205.0/72.0; c2 = 8.0/5.0; c3 = -1.0/5.0; c4 = 8.0/315.0; c5 = -1.0/560.0;

if nargout>2
    bc_top=zeros(5,nx,nt);
    bc_bottom=zeros(5,nx,nt);
    bc_left=zeros(nz,5,nt);
    bc_right=zeros(nz,5,nt);
end

% setup ABC and temperary variables
v=padvel(v,nbc); refl=padvel(refl,nbc);
abc=AbcCoef2D(v,nbc,dx);
alpha=(v*dt/dx).^2; kappa=abc*dt;
temp1=2+2*c1*alpha-kappa; temp2=1-kappa;
beta_dt = (v*dt).^2;
s=expand_source(s,nt);
[isx,isz,igx,igz]=adjust_sr(sx,sz,gx,gz,dx,nbc);

p1=zeros(size(v)); p0=zeros(size(v));
q1=zeros(size(v)); q0=zeros(size(v));

% Time Looping
for it=1:nt
    p=temp1.*p1-temp2.*p0+alpha.*...
        (c2*(circshift(p1,[0,1,0])+circshift(p1,[0,-1,0])+circshift(p1,[1,0,0])+circshift(p1,[-1,0,0]))...
        +c3*(circshift(p1,[0,2,0])+circshift(p1,[0,-2,0])+circshift(p1,[2,0,0])+circshift(p1,[-2,0,0]))...
        +c4*(circshift(p1,[0,3,0])+circshift(p1,[0,-3,0])+circshift(p1,[3,0,0])+circshift(p1,[-3,0,0]))...
        +c5*(circshift(p1,[0,4,0])+circshift(p1,[0,-4,0])+circshift(p1,[4,0,0])+circshift(p1,[-4,0,0])));
    p(isz,isx) = p(isz,isx) + beta_dt(isz,isx) * s(it);
    % dipole source
    %p(isz-2,isx) = p(isz-2,isx) - beta_dt(isz-2,isx) *wavelet(it);
    if isFS
        p(nbc+1,:)=0.0;
        p(nbc:-1:nbc-3,:) = - p(nbc+2:nbc+5,:);
    end
    if nargout>2 % save BC
        [bc_top(:,:,it),bc_bottom(:,:,it),bc_left(:,:,it),bc_right(:,:,it)]=save_boundary(p,nz,nx,nbc);
    end
    q=temp1.*q1-temp2.*q0+alpha.*...
        (c2*(circshift(q1,[0,1,0])+circshift(q1,[0,-1,0])+circshift(q1,[1,0,0])+circshift(q1,[-1,0,0]))...
        +c3*(circshift(q1,[0,2,0])+circshift(q1,[0,-2,0])+circshift(q1,[2,0,0])+circshift(q1,[-2,0,0]))...
        +c4*(circshift(q1,[0,3,0])+circshift(q1,[0,-3,0])+circshift(q1,[3,0,0])+circshift(q1,[-3,0,0]))...
        +c5*(circshift(q1,[0,4,0])+circshift(q1,[0,-4,0])+circshift(q1,[4,0,0])+circshift(q1,[-4,0,0])));
    q=pertubation(p1,q,refl,beta_dt);
    for ig=1:ng
        seis(it,ig)=q(igz(ig),igx(ig));
    end
    p0=p1; p1=p;
    q0=q1; q1=q;
end
if nargout>1 % save final wavefield
    bc_p_nt_1=p0;
    bc_p_nt=p1;
end
end