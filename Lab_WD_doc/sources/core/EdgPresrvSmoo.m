function B = EdgPresrvSmoo(A, W2)
% Edge-preserving smoothing, w/ smoothing width 2*W2+1
% along columns of A
% Ref
% [1] Luo, Y., et al 2002, Edge-preserving smoothing ...
[M,N] = size(A);
mu    = rect1Dconvs_nrmlz(A', W2, N, M);
mu2   = rect1Dconvs_nrmlz((A.*A)', W2, N, M);
vr    = mu2 - mu.*mu;
mu    = mu';      vr = vr';
B     = A;
for k = 1:N
  for j = 1:M
    hd  = max(1,j-W2);  tl = min(M,j+W2);
	  [mn, idx] = min(vr(hd:tl, k));
	  B(j,k)    = mu(hd+idx-1, k);
  end
end
return
%======================
N = 50;  W2=12;
A = [zeros(N,1); ones(N,1)] + 0.2*randn(2*N,1);
figure,
plot(A), hold on
B = EdgPresrvSmoo(A, W2);
plot(B,'g')

