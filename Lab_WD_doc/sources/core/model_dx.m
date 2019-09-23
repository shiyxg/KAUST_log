function [model_dx] = model_dx(model,dx);
model_dx = zeros(size(model));
[nz,nx] = size(model);
% 4 order accuracy
a = 1/12; b = -2/3 ; c = -b; d = -a;
for k = 1:nz
    for j = 3:nx-2        
        model_dx(k,j) = (a*model(k,j-2)+b*model(k,j-1) +c* model(k,j+1) +d* model(k,j+2))/(dx);
    end
end
model_dx(:,2) = model_dx(:,3); model_dx(:,1) = model_dx(:,2);
model_dx(:,nx-1) = model_dx(:,nx-2); model_dx(:,nx) = model_dx(:,nx-1);