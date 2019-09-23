%elastic image condition by zongcaf

function [cl_img,cm_img]=image_condition(cl_img,cm_img,xx_b,zz_b,xz_b,fux,fuz,bwx,bwz,pad_top,nbc,dt)

xx_b0=xx_b(pad_top+1:end-nbc,nbc+1:end-nbc);
zz_b0=zz_b(pad_top+1:end-nbc,nbc+1:end-nbc);
xz_b0=xz_b(pad_top+1:end-nbc,nbc+1:end-nbc);

% cl_img=cl_img-(fux+bwz).*(xx_b0+zz_b0)*dt;
% cm_img=cm_img-2*(xx_b0.*fux+zz_b0.*bwz+1/2*xz_b0.*(fuz+bwx))*dt;


 cl_img=cl_img+(fux+bwz).*(xx_b0+zz_b0);
 cm_img=cm_img+2*(xx_b0.*fux+zz_b0.*bwz+1/2*xz_b0.*(fuz+bwx));

end