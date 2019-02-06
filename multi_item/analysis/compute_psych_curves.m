% Draw psychometric curves of proportion response change as a function of
% magnitude of actual change (delta) and number of high-reliability items,
% as well as hit and false alarm rates as a function of number of
% high-reliability items. The first argument is the number of bins of delta
% to use, the next two are 1 if plotting subject and/or model data,
% respectively, and the last one is

function [delta_bin_vec,p_C_hat_mat_subj,HR_subj,FA_subj] = compute_psych_curves(num_delta_bins,plot_subj_data,plot_model_data,Model_Data_Cell)

if nargin==3 % no model data
    plot_model_data = 0;
end

% Get P_C_hat = 1 as a function of delta

% first bin will contain only delta == 0 points
delta_bin_vec = linspace(eps,pi/2+eps,num_delta_bins);

% load subject data
load Subj_data_cell.mat
Subj_data_cell = revert_data(Subj_data_cell);
num_subj = length(Subj_data_cell);

% get subject data and model curves
% [p_C_hat_mat_subj,curr_mean_subj,curr_stderr_subj,FA_mean_subj,FA_stderr_subj,HR_mean_subj,HR_stderr_subj,N_vec] = get_curves(delta_bin_vec,Subj_data_cell);
[p_C_hat_mat_subj,HR_subj,FA_subj,N_vec] = get_curves(delta_bin_vec,Subj_data_cell);

if plot_subj_data
    
    curr_mean_subj = squeeze(mean(p_C_hat_mat_subj,1));
    curr_stderr_subj = squeeze(std(p_C_hat_mat_subj,[],1)/sqrt(num_subj));
    
%     close
    figure
    subplot(1,2,1)
%     stdlines = [0 0 1;.9 .6 0;0 1 0;1 0 0; 0 1 1];
    for i = 1:length(N_vec)
        %         errorbar(delta_bin_vec,curr_mean_subj(i,:),curr_stderr_subj(i,:),'Color',stdlines(i,:),'LineWidth',1.5);
        errorbar(delta_bin_vec,curr_mean_subj(i,:),curr_stderr_subj(i,:),'LineWidth',1.5);
        hold on
    end
    legend([repmat('N_H=',length(N_vec),1) num2str(N_vec)],'Location','SouthEast')
    title('Probability report "Change"')
    xlabel('Magnitude of change in radians')
    defaultplot
    
    subplot(1,2,2)
    
    HR_point_color = [0 0 0];
    FA_point_color = (1/255)*[148 138 84];
    
    FA_mean_subj = mean(FA_subj,1);
    FA_stderr_subj = std(FA_subj,[],1)/sqrt(num_subj);
    HR_mean_subj = mean(HR_subj,1);
    HR_stderr_subj = std(HR_subj,[],1)/sqrt(num_subj);
    
    hold on
    errorbar(N_vec,HR_mean_subj,HR_stderr_subj,'Color',HR_point_color,'LineWidth',1.5);
    errorbar(N_vec,FA_mean_subj,FA_stderr_subj,'Color',FA_point_color,'LineWidth',1.5);
    legend('Hit Rate','False Alarm Rate')
    title('Hit and false alarm rates')
    ylim([0 1]);
    xlabel('Number of high reliability items (N_H)')
    set(gcf,'OuterPosition',[0 0 1000 500])
    title('Subject data')
    defaultplot
end

if nargin == 4 % if given the model predictions
    [p_C_hat_mat,HR,FA,N_vec] = get_curves(delta_bin_vec,Model_Data_Cell);
%     [p_C_hat_mat,curr_mean,curr_stderr,FA_mean,FA_stderr,HR_mean,HR_stderr,N_vec] = get_curves(delta_bin_vec,Model_Data_Cell);    

    % plotting model predictions as well
    if plot_model_data
        
        
        % plotting stuff
        figure
        stdcols = [1 .8 .5;1 .8 .8; .8 1 .8; .8 .8 1; .5 .8 1];
        subplot(1,2,1)
        hold on
        
        % get the model curves
        polyX = [delta_bin_vec delta_bin_vec(end:-1:1)];
        
        p_C_hat_mat_mean = squeeze(mean(p_C_hat_mat,1));
        p_C_hat_mat_stderr = squeeze(std(p_C_hat_mat,[],1)/sqrt(num_subj));
        FA_mean = mean(FA,1);
        FA_stderr = std(FA,[],1)/sqrt(num_subj);
        HR_mean = mean(HR,1);
        HR_stderr = std(HR,[],1)/sqrt(num_subj);
        
        for i = 1:length(N_vec)
            polyY(1:length(delta_bin_vec)) = p_C_hat_mat_mean(i,:)-p_C_hat_mat_stderr(i,:);
            polyY((2*length(delta_bin_vec)):-1:(length(delta_bin_vec)+1)) = p_C_hat_mat_mean(i,:)+p_C_hat_mat_stderr(i,:);
            fill(polyX,polyY,stdcols(i,:),'LineStyle','none');
        end
        
        % plot the subject data for p_resp_change
        stdlines = [.9 .6 0;1 0 0;0 1 0;0 0 1; .25 .4 .5];
        hold on
        for i = 1:length(N_vec)
            errorbar(delta_bin_vec,curr_mean_subj(i,:),curr_stderr_subj(i,:),'o','Color',stdlines(i,:),'LineWidth',2);
        end
        legend([repmat('N=',length(N_vec),1) num2str(N_vec)],'Location','SouthEast')
        xlim([-.05 (pi/2+.05)])
        ylim([0 1])
        set(gca,'YTick',[.2 .4 .6 .8 1]);
        set(gca,'XTick',[0 pi/6 pi/3 pi/2]);
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        title('Probability report "Change"')
        xlabel('Magnitude of change in radians')
        
        % HR and FA
        subplot(1,2,2)
        hold on
        
        % get model data
        polyX = [N_vec' N_vec(end:-1:1)'];
        HR_fill_color = 191/255*[1 1 1];
        HR_point_color = [0 0 0];
        FA_fill_color = (1/255)*[221 217 195];
        FA_point_color = (1/255)*[148 138 84];
        
        polyY_HR(1:length(N_vec)) = HR_mean-HR_stderr;
        polyY_HR((2*length(N_vec)):-1:(length(N_vec)+1)) = HR_mean+HR_stderr;
        fill(polyX,polyY_HR,HR_fill_color,'LineStyle','none');
        
        polyY_FA(1:length(N_vec)) = FA_mean-FA_stderr;
        polyY_FA((2*length(N_vec)):-1:(length(N_vec)+1)) = FA_mean+FA_stderr;
        fill(polyX,polyY_FA,FA_fill_color,'LineStyle','none');
        
        % plot subject p_resp_change
        errorbar(N_vec,HR_mean_subj,HR_stderr_subj,'o','Color',HR_point_color,'LineWidth',2);
        errorbar(N_vec,FA_mean_subj,FA_stderr_subj,'o','Color',FA_point_color,'LineWidth',2);
        xlim([(min(N_vec)-1) (max(N_vec)+1)])
        set(gca,'YTick',[.2 .4 .6 .8 1]);
        set(gca,'XTick',N_vec);
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        ylim([0 1])
        legend('Hit Rates','False Alarms')
        title('Hit and false alarm rates')
        
        xlabel('Number of high reliability items (N_H)')
        
        set(gcf,'OuterPosition',[0 0 1000 500])
        title('Model predictions on subject data')        
    end
end

% function [p_C_hat_mat,curr_mean,curr_stderr,FA_mean,FA_stderr,HR_mean,HR_stderr,N_vec] = get_curves(delta_bin_vec,curr_data_cell)
function [p_C_hat_mat,HR,FA,N_vec] = get_curves(delta_bin_vec,curr_data_cell)

num_subj = length(curr_data_cell);
num_delta_bins = length(delta_bin_vec);

% intialize matrices for holding statistics
rels = unique(curr_data_cell{1}(:,39:54));
N_vec = unique(sum(curr_data_cell{1}(:,39:42)==rels(2),2));
p_C_hat_mat = zeros(num_subj,length(N_vec),length(delta_bin_vec));
[HR, FA] = deal(zeros(num_subj,length(N_vec)));

for subj_idx = 1:num_subj
    
    % get current subject data
    curr_data = curr_data_cell{subj_idx};
    rels = unique(curr_data(:,39:42));
    N_high_vec = sum(curr_data(:,39:42)==rels(2),2);
    N_vec = unique(N_high_vec);
    
    curr_data_delta = 0.5*sum(abs(circ_dist((pi/90)*curr_data(:,56:63),(pi/90)*curr_data(:,64:71))),2);
    
    % loop over set size
    for N_idx = 1:length(N_vec)
        
        curr_N = N_high_vec == N_vec(N_idx);
        curr_N_data = curr_data(curr_N,:);
        curr_N_delta = curr_data_delta(curr_N);
        
        % first data point contains all no-change stimuli
        resp_C = mean(curr_N_data((curr_N_delta<eps),2),1);
        p_C_hat_mat(subj_idx,N_idx,1)= resp_C;
        
        % loop over deltas
        for delta_idx = 2:num_delta_bins
            
            resp_C = mean(curr_N_data((curr_N_delta>delta_bin_vec(delta_idx-1)) & ...
                (curr_N_delta<=delta_bin_vec(delta_idx)),2),1);
            
            p_C_hat_mat(subj_idx,N_idx,delta_idx)= resp_C;
            
        end
        
        % compute hit rate and false alarm rate
        HR(subj_idx,N_idx) = sum(curr_N_data((curr_N_data(:,1)~=0),2),1)/sum((curr_N_data(:,1)~=0),1);
        FA(subj_idx,N_idx) = sum(curr_N_data((curr_N_data(:,1)==0),2),1)/sum((curr_N_data(:,1)==0),1);
        
    end
    
end


