function [g_cl_con,g_cm_con]=preconditioning(g_cl,g_cm,g_illum)
%fpr preconditioning by Zongcai Feng  2015.4.28

%Illumination Preconditioning
g_cl_con=g_cl./g_illum;
g_cm_con=g_cm./g_illum;

end

