figure;
x0=10;
y0=10;
width=400;
height=300;
set(gcf,'position',[x0,y0,width,height])

plot(bert_oodomain(:,1),bert_oodomain(:,2), '-^r','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','red');

xlabel('Number of augmented context paragraph (n)','FontSize',15, 'FontName', 'Times');
ylabel('Performance Gain (%)', 'FontSize',15, 'FontName', 'Times');

xlim([0,16]);
ylim([0,20]);
xticks([0 4 8 16]);
yticks([0 4 8 12 16 20]);

ax = gca;
ax.YAxis.FontSize = 15
ax.XAxis.FontSize = 15

ax.XAxis.FontName = 'Times'
ax.YAxis.FontName = 'Times'
grid on




% text(-29,-2.4*1e-5,'Device C','FontSize',18,'FontWeight','bold');
