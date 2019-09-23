function [Px,Pz,Px_dm,Pz_dm] = model_smoothing(v,dx);

[nz,nx] = size(v);
Px_dm = zeros(size(v));
Pz_dm = zeros(size(v));

for k = 1:nz
    for j = 2:nx-1
         
    Px_dm(k,j) = (-2*v(k,j) + v(k,j-1) + v(k,j+1))/(dx*dx);     
        
    end
end

Px_dm(:,1) = Px_dm(:,2);  Px_dm(:,nx) = Px_dm(:,nx-1);


for k = 2:nz-1
    for j = 1:nx
         
    Pz_dm(k,j) = (-2*v(k,j) + v(k-1,j) + v(k+1,j))/(dx*dx);     
        
    end
end

Pz_dm(1,:) = Pz_dm(2,:);  Pz_dm(nz,:) = Pz_dm(nz-1,:);

v_dz = model_dz(v,dx);
v_dx = model_dx(v,dx);
Px = v_dx.^2;
Pz = v_dz.^2;
