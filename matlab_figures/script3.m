figure;
x0=10;
y0=10;
width=400;
height=300;
set(gcf,'position',[x0,y0,width,height])

plot(alation_study_SR(:,1),alation_study_SR(:,2), '-^b','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','blue');
% hold on;
% plot(alation_study_SR(:,1),alation_study_SR(:,3), '-^r','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','red');
% hold on;
% plot(alation_study_SR(:,1),alation_study_SR(:,4), '-^c','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','cyan');
% hold on;
% plot(alation_study_SR(:,1),alation_study_SR(:,5), '-^k','LineWidth',2,'MarkerSize',12,'MarkerEdgeColor','black');

xlabel('α_{SR}','FontSize',15, 'FontName', 'Times');
ylabel('Performance Gain (%)', 'FontSize',15, 'FontName', 'Times');

xlim([0,0.5]);
ylim([0,9]);
xticks([0 0.05 0.1 0.2 0.3 0.4 0.5]);
% yticks([0 4 8 12 16 20]);

ax = gca;
ax.YAxis.FontSize = 15
ax.XAxis.FontSize = 15

ax.XAxis.FontName = 'Times'
ax.YAxis.FontName = 'Times'
grid on

text(0.23,9.5,'SR','FontSize',18,'FontWeight','bold');

% legend("SR", "RI","RS","RD")
% lgd = legend;
% lgd.FontSize = 14;