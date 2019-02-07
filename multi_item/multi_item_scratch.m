
%% set up data into useable format

subj_ID_cell = {'POO'};
pres2stimuli_cell = {'Ellipse'};
Subj_data_cell = combine_all_data(subj_ID_cell,pres2stimuli_cell);

save('Subj_data_cell.mat','Subj_data_cell')

%% plot psychometric function of current dataset

% close all
nBins = 6;
plot_subjdata = 1;
[delta_bin_vec,p_C_hat_mat_subj,HR_subj,FA_subj] = compute_psych_curves(nBins,plot_subjdata);

% plot indvl psychometric curves
nSubj = size(FA_subj,1);
N_vec = (0:4)';


% plot
figure
for isubj = 1:nSubj;
    
    % proportion report change
    subplot(nSubj,2,2*isubj-1)
    hold on;
    for irel = 1:length(N_vec);
        plot(delta_bin_vec,squeeze(p_C_hat_mat_subj(isubj,irel,:)));
    end
    ylim([0 1]);
    if isubj == nSubj; 
        xlabel('Magnitude of change in radians')
        legend([repmat('N_H=',length(N_vec),1) num2str(N_vec)])
    end;
    if isubj == 1; title('Probability report "Change"'), end;
    defaultplot
    
    % HR and FA
    subplot(nSubj,2,2*isubj)
    hold on; 
    plot(N_vec,HR_subj(isubj,:)); % hit rate
    plot(N_vec,FA_subj(isubj,:)); % false alarm
    ylim([0 1]);
    if (isubj == 1); title('Hit and false alarm rates'), end
    if (isubj == nSubj);
        legend('Hit Rate','False Alarm Rate')
        xlabel('Number of high reliability items (N_H)')
    end
    defaultplot
end

%% fit models
clear all

subjid = 'POO';
pres2stimuli = 'Ellipse';

model_mat = ...
    [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ... % VPO model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ... % VPM model variants
    2 2 1 1;  2 3 1 1; 2 4 1 1; ...  % EPO model variants
    2 2 2 1;  2 3 2 1; 2 4 2 1]; % EPM model variants

for imodel = 2:size(model_mat,1)
    run_model_reliability(subjid, pres2stimuli, model_mat(imodel,:));
end


%% model comparson
clear all

subj_id_cell = {'POO'};
pres2stimuli = 'Ellipse';

model_mat = ...
    [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ... % VPO model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ... % VPM model variants
    2 2 1 1;  2 3 1 1; 2 4 1 1; ...  % EPO model variants
    2 2 2 1;  2 3 2 1; 2 4 2 1]; % EPM model variants

nModels = size(model_mat,1);
nSubj = length(subj_id_cell);

LLMat = nan(nSubj,nModels);
for imodel = 1:nModels
    currmodel = model_mat(imodel,:);
    LLMat(:,imodel) = compute_BMC(currmodel,subj_id_cell,pres2stimuli);
end
