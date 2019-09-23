function [model_dz] = model_dz(model,dz);
model_dz = zeros(size(model));
[nz,nx] = size(model);

% 4 order accuracy
a = 1/12; b = -2/3 ; c = -b; d = -a;

for k = 3:nz-2
    for j = 1:nx        
        model_dz(k,j) = (a*model(k-2,j)+b*model(k-1,j) +c* model(k+1,j) +d* model(k+2,j))/(dz);
    end
end

model_dz(nz-1,:) = model_dz(nz-2,:); model_dz(nz,:) = model_dz(nz-1,:);
model_dz(2,:) = model_dz(3,:); model_dz(1,:) = model_dz(2,:);