function plotit(dx,p2,nx,nz,it,dt) 
figure(7); 
imagesc((1:nx),(1:nz),100*p2);
title(['SNAPSHOT  T= ',num2str(dt*it),' sec']);
xlabel('Horizontal Offset (m)');ylabel('Depth (m)');
colorbar;
pause(1)
