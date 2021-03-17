clear; clc; close all;
figure;
x0=10;
y0=10;
width=400;
height=300;
set(gcf,'position',[x0,y0,width,height])

x = [0,1,2,3];
y = [47.32, ];
plot(bert_indomain_oodomain(:,1),bert_indomain_oodomain(:,2), '-^g','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','red');

xlabel('Number of re-initialization layers (\itL\rm)','FontSize',15, 'FontName', 'Times');
ylabel('F1 score', 'FontSize',15, 'FontName', 'Times');
xlim([0,3]);
ylim([45,50]);
xticks([0 1 2 3]);
% yticks([0 4 8 12 16 20]);

ax = gca;
ax.YAxis.FontSize = 15;
ax.XAxis.FontSize = 15;
ax.XAxis.FontName = 'Times';
ax.YAxis.FontName = 'Times';
grid on