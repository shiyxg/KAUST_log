function tau = FirstBreakPicking(csg, T)
% csg   nt x ng  (t increases from [1 --> nt] dt);  ng: # geophones
% tau    1 x ng   1st arrival times, in dt
%  T     1 x 1    period (in terms of # time steps)
% Ref:
% [1] Sabbione, Automatic first-breaks picking, Geophysics, 2010
% 'Modified Coppens' Method'

beta = 0.2;
ne2  = floor(1.5*T/2);  % edg-presrv-smoothing width:  2ne2 + 1
nl   = ceil(T);
% 1. normalize to peak per trace:
mx   = max(abs(csg),[],1);
[nt, ng] = size(csg);
csg  = csg ./ mx(ones(nt,1),:);
csg  = csg .* csg;
% 2. compute energy in two windows, & their ratio:
E2 = cumsum(csg,1);
E1 = E2;
E1(nl+1:nt,:) = E2(nl+1:nt,:) - E2(1:nt-nl,:);
ER = E1./(E2 + beta);
% 3. EPS to identify the break
smo= EdgPresrvSmoo(ER, ne2);
edg= diff(smo,1,1);
[mx, tau] = max(edg,[],1);
% plot(ER,'r'), hold on, plot(smo,'g'), plot(edg,'k')

return
% %==============
% A = [[zeros(50,1);  randn(50,1)],  [zeros(20,1);  randn(80,1)]];
% figure, plot(A), hold on
% tau = FirstBreakPicking(A, 10)
