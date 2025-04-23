%%
clc;
clear;
close;
Data_rew = readtable('run-.-tag-rollout_ep_rew_mean.csv','ReadVariableNames',false);
Data_len = readtable('run-.-tag-rollout_ep_len_mean.csv','ReadVariableNames',false);
Data_exr = readtable('run-.-tag-rollout_exploration_rate.csv','ReadVariableNames',false);
Data_loss = readtable('run-.-tag-train_loss.csv','ReadVariableNames',false);

step_rew = table2array(Data_rew(:,2));
value_rew = table2array(Data_rew(:,3));

step_len = table2array(Data_len(:,2));
value_len = table2array(Data_len(:,3));

step_exr = table2array(Data_exr(:,2));
value_exr = table2array(Data_exr(:,3));

step_loss = table2array(Data_loss(:,2));
value_loss = table2array(Data_loss(:,3));


LineWidth = 2;


subplot(2, 2, 1);
plot(step_rew, value_rew, 'LineWidth', LineWidth)
set(ylabel('Episode length'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('Mean episode reward'),'interpreter','latex');
grid on
grid minor

subplot(2, 2, 2);
plot(step_len, value_len, 'LineWidth', LineWidth)
set(ylabel('Episode length'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('Mean episode length'),'interpreter','latex');
grid on
grid minor

subplot(2, 2, 3);
plot(step_exr, value_exr, 'LineWidth', LineWidth)
set(ylabel('Episode length'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('Exploration Rate'),'interpreter','latex');
grid on
grid minor

subplot(2, 2, 4);
plot(step_loss, value_loss, 'LineWidth', LineWidth)
set(ylabel('Episode length'),'interpreter','latex');
set(xlabel('Step'),'interpreter','latex');
set(title('Train Loss'),'interpreter','latex');
grid on
grid minor

set(sgtitle('Torque-based energy consumption without continuation after lost detection'),'interpreter', 'latex')

%%

AspectRatio = 4; %4 Recommended (single column), 2.2 for dual column
PlotSize = 6; % This is how large the plot is compared to the text 

ExportPlot(PlotSize,AspectRatio,'C:\Users\adam\Desktop\log_to_analize')
function ExportPlot(PlotSize,AspectRatio,Path)
    addToolbarExplorationButtons(gcf);
    set(gcf,'units','centimeters','color','white') %Remove grey background
    Pgcf = get(gcf,'position'); %Get default position
    set(gcf,'position',[Pgcf(1:2),PlotSize*AspectRatio,PlotSize])
    % print(Path,'-dpdf') %Remove upper and right margins.
end
