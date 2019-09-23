figure
 figure_FontSize=16;
 set(gca, 'Fontname', 'Arial', 'Fontsize', 18);
subplot(221);imagesc(vs_d);title('Vs True model','fontsize',16);colorbar;
xlabel('Distance (m)');
ylabel('Depth (m)');

 set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
 set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
 set(findobj('FontSize',16),'FontSize',figure_FontSize);
subplot(222);imagesc(vs);title('Vs Intial model','fontsize',16);colorbar;
xlabel('Distance (m)');
ylabel('Depth (m)');
%  figure_FontSize=16;
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
% set(findobj('FontSize',16),'FontSize',figure_FontSize);
subplot(223);imagesc(vs1);title('Vs Inversion Resuelt','fontsize',16);colorbar;
xlabel('Distance (m)');
ylabel('Depth (m)');
%  figure_FontSize=16;
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
% set(findobj('FontSize',16),'FontSize',figure_FontSize);
subplot(224);imagesc(dk_vs);title('Vs Gradient Snapshot','fontsize',16);colorbar;
xlabel('Distance (m)');
ylabel('Depth (m)');
%  figure_FontSize=16;
% set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
% set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
% set(findobj('FontSize',16),'FontSize',figure_FontSize);
%imagesc(vs_d);figure(gcf);


%这4句是将字体大小改为8号字，在小图里很清晰


% set(gcf,'Position',[100 100 260 220]);
% %这句是设置绘图的大小，不需要到word里再调整大小。我给的参数，图的大小是7cm
% 
% set(gca,'position',[0.14 0.3 0.8 0.5])
%这句是设置xy轴在图片中占的比例，可能需要自己微调。

%set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);

%这句是将线宽改为2




%设置图片的字体类型和字号大小的。
print 2.eps -depsc2 -r600