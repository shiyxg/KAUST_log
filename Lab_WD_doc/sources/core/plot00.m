%Plot figure for the surface wave inversion
%JIng LI kaust



figure_FontSize=12;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',12),'FontSize',figure_FontSize);
set(gcf,'Position',[100 100 260 220]);
%这句是设置绘图的大小，不需要到word里再调整大小。我给的参数，图的大小是7cm

set(gca,'Position',[.13 .17 .80 .74]);
%这句是设置xy轴在图片中占的比例，可能需要自己微调。

set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);

%这句是将线宽改为2

set(gca, 'Fontname', 'Times newman', 'Fontsize', 12);
figure
subplot(2,2,1);
% plotopt = [1 50 0.01];
imagesc(vs_d);axis image ; 
%caxis([-plotopt(3) plotopt(3)]);

subplot(2,2,2);
% plotopt = [1 50 0.01];
imagesc(vs);axis image ;

subplot(2,2,3);
% plotopt = [1 50 0.01];
imagesc(vs1);axis image ;