clc;
clear;
close;
Data1 = readtable('run-.-tag-rollout_ep_len_mean.csv','ReadVariableNames',false);
%Data2 = readtable('bounding_box_piecewise_linear_ep_len_mean.csv','ReadVariableNames',false);

step1= Data1(:,2);
value1= Data1(:,3);

% step2= Data2(:,2);
% value2= Data2(:,3);

step1= table2array(step1);
value1= table2array(value1);

% step2= table2array(step2);
% value2= table2array(value2);

AspectRatio = 4; %4 Recommended (single column), 2.2 for dual column
PlotSize = 6; % This is how large the plot is compared to the text 

plot(step1, value1)
% hold on
% plot(step2, value2)

set(ylabel('Episode length'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('Mean episode length'),'interpreter','latex');
% set(legend('Depth camera','Bounding box estimation','Location','northeast'),'interpreter','latex');

grid on
grid minor

ExportPlot(PlotSize,AspectRatio,'D:\Master first semester\object detection model')
function ExportPlot(PlotSize,AspectRatio,Path)
addToolbarExplorationButtons(gcf);
set(gcf,'units','centimeters','color','white') %Remove grey background
Pgcf = get(gcf,'position'); %Get default position
set(gcf,'position',[Pgcf(1:2),PlotSize*AspectRatio,PlotSize])
print(Path,'-dpdf') %Remove upper and right margins.
end
