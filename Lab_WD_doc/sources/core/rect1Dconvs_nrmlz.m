function B = rect1Dconvs_nrmlz(A, W2, M, N)
% A    M X N
% W2   M X 1    A's kth row conv. w/ rect(2*W2[k]+1), and then
%               divided by # of summing terms, so the output is normalized.
%               if W2 is scalar, then all conv kernels are of equal width.
% B    M X N    the averaged output

B    = zeros(M,N);
narr = (N-1) <= W2;  % narrow
if numel(W2) > 1
    if any(narr)
        B(narr,1) = sum(A(narr,:),2)./N;  % normalize
        B(narr,:) = B(narr, ones(1,N));
    end
    if any(~narr)
        B(~narr,:) = cumsum(A(~narr,:), 2);
        slp   = 1:N;
        for k = find(~narr)'  % important! must be a row vector!
            idxl  = [1+W2(k):N, N(ones(1,W2(k)))];
            idxr  = 1: N-W2(k)-1;
            padrl = zeros(1,W2(k)+1);
            B(k,:)= (B(k,idxl)-[padrl,B(k,idxr)])./(slp(idxl)-[padrl,slp(idxr)]);
        end
    end
else        % scalar W2
    if narr
        B(:,1) = sum(A,2)./N;           % normalize
        B      = B(:, ones(1,N));
        return
    end
    slp = 1:N;
    B   = cumsum(A, 2);
    idxl= [1+W2:N, N(ones(1,W2))];
    idxr= 1: N-W2-1;
    padrl = zeros(1,W2+1);
    Z   = slp(idxl) -[padrl, slp(idxr)];
    B   = (B(:,idxl)-[padrl(ones(M,1),:),B(:,idxr)])./Z(ones(M,1),:);
end
return
%=============
% M = 4;  N = 10;
% W2= [12; 10; 3; 2];
% A = randn(M, N);
% B = rect1Dconvs_nrmlz(A, W2, M, N);
% sum(A(1,:))/N - B(1,5)
% sum(A(2,:))/N - B(2,end)
% sum(A(3,5-W2(3):5+W2(3)))/(2*W2(3)+1) - B(3,5)
% sum(A(4,end-W2(4):end))/(W2(4)+1) - B(4,end)
% tested!

%% [0 1 0 0 0 0 0] (*) [1 1 1] normalize
% --> [0.5, 0.33, 0.33, 0, 0, ..]
% that is, the peak location is shifted.
% Alternative:
% (1) pad the boundary w/ boundary elements,
% (2) conv w/ rect
% (3) divided by a constant normalization factor.


