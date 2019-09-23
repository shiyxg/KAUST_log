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


%��4���ǽ������С��Ϊ8���֣���Сͼ�������


% set(gcf,'Position',[100 100 260 220]);
% %��������û�ͼ�Ĵ�С������Ҫ��word���ٵ�����С���Ҹ��Ĳ�����ͼ�Ĵ�С��7cm
% 
% set(gca,'position',[0.14 0.3 0.8 0.5])
%���������xy����ͼƬ��ռ�ı�����������Ҫ�Լ�΢����

%set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);

%����ǽ��߿��Ϊ2




%����ͼƬ���������ͺ��ֺŴ�С�ġ�
print 2.eps -depsc2 -r600