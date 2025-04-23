
clc;
clear;
close;
Data1 = readtable('run-.-tag-train_loss.csv','ReadVariableNames',false);
step1= Data1(:,2);
value1= Data1(:,3);

step1= table2array(step1);
value1= table2array(value1);
AspectRatio = 4; %4 Recommended (single column), 2.2 for dual column
PlotSize = 6; % This is how large the plot is compared to the text 

plot(step1, value1)

set(ylabel('Value'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('$Train\ loss$'),'interpreter','latex');
%set(legend('Model1','Location','northeast'),'interpreter','latex');
%,'Model2','Model3','Model4'
%axis([time(1) time(end) 20 100])
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
