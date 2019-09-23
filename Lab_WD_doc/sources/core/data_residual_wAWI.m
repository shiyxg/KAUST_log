function [ data_res,delta ] = data_residual_wAWI(seis,csg_true,method,tw,level)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%  seis   :  d_obs
%  csg_true: d_cal
%  method : 1 t -x domain
%         : 2 t -x domain, cross-correlations
%         : 3 f -k domain  : AWI
%         : 4 f -k domain (not working)  : w-AWI, divide gather into sub-section with tw (tw: trace window);with overlaping%         
%        nw:   number of sub-windows 

% seis = seismo_w;

[nt,ng] = size(seis);

switch method
    case 1
    data_res=seis-csg_true; 
    delta = data_res;
   
    case 2    
    data_res = zeros(size(seis));
    p_nor =  zeros(size(seis));
    d_nor =  zeros(size(seis));    
    p = seis;
    d = csg_true;
    for k = 1:ng 
        p_nor(:,k) = p(:,k)./norm(p(:,k));   
        d_nor(:,k) = d(:,k)./norm(d(:,k)); 
        data_res(:,k) = ((dot(p_nor(:,k),d_nor(:,k))*p_nor(:,k) - d_nor(:,k)));
    end

    case 3
%         F_seis = fftshift(fft2(seis));
%         F_csg_true = fftshift(fft2(csg_true));
%         F_res = (F_seis./abs(F_seis)).*(abs(F_seis) - abs(F_csg_true));
%         data_res = real(ifft2(ifftshift(F_res)));
%         delta = abs(F_seis) - abs(F_csg_true);
        F_seis = (fft2(seis));
        F_csg_true = (fft2(csg_true));
        F_res = (F_seis./abs(F_seis)).*(abs(F_seis) - abs(F_csg_true));
        data_res = real(ifft2((F_res)));
        delta = abs(F_seis) - abs(F_csg_true);

    case 4        
        nw = ng - tw + 1;
        window = zeros(nt,ng,nw);
        sub_seis = zeros(nt,ng,nw);
        sub_csg_true = zeros(nt,ng,nw);
        for k = 1:nw
            for j = 1:ng
                window(:,k:tw+k-1,k) = 1;
            end
            sub_seis(:,:,k) = seis.*window(:,:,k);
            sub_csg_true(:,:,k) = csg_true.*window(:,:,k);
%             figure;subplot(121);imagesc(sub_seis(:,:,k));subplot(122);imagesc(sub_csg_true(:,:,k) );
            F_sub_seis(:,:,k) = fft2(sub_seis(:,:,k));
            F_sub_csg_true(:,:,k) = fft2(sub_csg_true(:,:,k));
%             figure;subplot(121);imagesc(abs(F_sub_seis(:,:,k)));subplot(122);imagesc(abs(F_sub_csg_true(:,:,k)) );
            Rr(:,:,k) = (F_sub_seis(:,:,k)./(abs(F_sub_seis(:,:,k))+0.0001*max(max(abs(F_sub_seis(:,:,k)))))).*(abs(F_sub_seis(:,:,k))-abs(F_sub_csg_true(:,:,k)));
%             Rr(:,:,k) = (F_sub_seis(:,:,k)./(abs(F_sub_seis(:,:,k)))).*(abs(F_sub_seis(:,:,k))-abs(F_sub_csg_true(:,:,k)));
            
            res0 = real(ifft2(Rr(:,:,k)));     
            res(:,:,k) = res0.*window(:,:,k);
            
            delta(:,:,k) = abs(F_sub_seis(:,:,k))-abs(F_sub_csg_true(:,:,k));
        end
        data_res = sum(res,3);
        delta = sum(delta);
        
        case 5
            th = th_schedule0(3,csg_true,99,1,100);  
            N = [ 2 3 4 6 10 20 40 80]; 

            F_seis = fftshift(fft2(seis)); 
            A_seis = abs(F_seis); 
            Angle_seis = angle(F_seis);
            I = find(A_seis<th(N(level))); 
            A_seis(I)=0; 
            Y_seis = A_seis.*exp(1i*Angle_seis);                

            F_csg_true = fftshift(fft2(csg_true)); 
            A_csg_true = abs(F_csg_true); 
            Angle_csg_true = angle(F_csg_true);
            I = find(A_csg_true<th(N(level))); 
            A_csg_true(I)=0; 
            Y_csg_true = A_csg_true.*exp(1i*Angle_csg_true);

            F_res = (F_seis./(abs(F_seis)+0.0001*max(max(abs(F_seis))))).*(abs(Y_seis) - abs(Y_csg_true));
            data_res = real(ifft2(ifftshift(F_res)));
        
        
        
end







