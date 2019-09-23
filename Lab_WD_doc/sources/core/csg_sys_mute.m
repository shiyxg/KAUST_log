function [ seismo_w_mute] = csg_sys_mute(seismo_w,T,T1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% T = 10; used in picking the first arrival
% T1  period (points) for first arrival waveform
%       tau_p = FirstBreakPicking(seismo_w, T);
% tau_o = FirstBreakPicking(seismo_w, 4);
 [C, tau_o] = max(abs(seismo_w),[],1);
% [C, tau_o] = max((seismo_w),[],1);

%       figure;imagesc(seismo_w);hold on; plot(tau_p);
% figure;imagesc(seismo_w);hold on; plot(tau_o);
 

[nt,ng] = size(seismo_w);

tapering = zeros(size(seismo_w));    
for k = 1:ng
    
    b1 = tau_o(k) - 0.5*T1;
    b2 = tau_o(k) + 0.5*T1; % window size 
    
%   for j = 1:nt
%       if j< b1
%           tapering(j,k) = 0;
%       elseif j>=b1 && j<=tau_o(k);
%           tapering(j,k) = exp(-(j-tau_o(k))^2/2*(2*T)^2) ;
%       elseif j>=tau_o(k) && j<= b2;
%           tapering(j,k) = 1;
%       else
%           tapering(j,k) = exp(-(j-b2)^2/2*(2*T)^2) ;
%       end
%   end
    for j = 1:nt
          if j< b1-10*T
              tapering(j,k) = 0;
          elseif j>=b1-20*T && j<=b1;
              tapering(j,k) = exp(-(j-tau_o(k))^2/2*(2*T)^2) ;
          elseif j>=b1 && j<= b2;
              tapering(j,k) = 1;
          else
              tapering(j,k) = exp(-(j-b2)^2/2*(2*T)^2) ;
          end
    end
  
end
          
% csg_true_mute = csg_true.*tapering;
seismo_w_mute = seismo_w.*tapering;     
      
      

end

