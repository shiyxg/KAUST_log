function [ dk_cl,dk_cl_old,dk_cm,dk_cm_old,g_cl_old,g_cm_old ] = ...
    conjugate_gradiend( dk_cl_old,dk_cm_old,g_cl,g_cl_old,g_cm,g_cm_old,g_illum,k )
%algorithm see Schuster book P310, Q-LSRTM


[g_cl_con,g_cm_con]=preconditioning(g_cl,g_cm,g_illum);
[g_cl_old_con,g_cm_old_con]=preconditioning(g_cl_old,g_cm_old,g_illum);

%calculate beta
if k==1
    beta=0;
else
    beta=sum(g_cl(:).*g_cl_con(:)+g_cm(:).*g_cm_con(:))...
        /sum(g_cl_old(:).*g_cl_old_con(:)+g_cm_old(:).*g_cm_old_con(:));
end

dk_cl=g_cl_con+beta*dk_cl_old;
dk_cm=g_cm_con+beta*dk_cm_old;

g_cl_old=g_cl;
g_cm_old=g_cm;
dk_cl_old=dk_cl;
dk_cm_old=dk_cm;

end
