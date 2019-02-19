% Generate model predictions for each model and subject. This could all be 
% done completely in parallel per subject and model. 

fprintf('In this example analysis, models discussed in the paper\n') 
fprintf('models are fitted to the data of all subjects.\n')
fprintf('The entire analysis can take between 1-7 hours time, \n');
fprintf('depending on the speed of your computer, the number of \n')
fprintf('MC samples, and whether you parallelize "for" loops.\n\n');
fprintf('Press any key to start the analysis.\n');
pause()

% addpath(genpath('../'))
load Subj_data_cell.mat

% set random seed
rng('shuffle')

% set the model indices; same order as in Table 4 in the paper
model_mat = ...
    [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ... % VPO model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ... % VPM model variants
    2 2 1 1;  2 3 1 1; 2 4 1 1; ...  % EPO model variants
    2 2 2 1;  2 3 2 1; 2 4 2 1]; % EPM model variants
subjid = 'POO';
pres2stimuli = 'Ellipse';
for imodel = 2:size(model_mat,1)
    run_model_reliability(subjid, pres2stimuli, model_mat(imodel,:));
end

% Get the model predictions and log likelihoods. This takes a LONG time. I
% suggest using matlab parallel toolbox to parallelize the inner for-loop,
% in addition to possibly running the different models (first for-loop) on
% different CPUs simultaneously.
for model_num = 1:size(model_mat,1)
    for subj_num = 1:length(Subj_data_cell)
        fprintf(['\nModel ' num2str(model_mat(model_num,:)) ', subject ' num2str(subj_num) '\n\n']);
        run_model_reliability(subj_num,model_mat(model_num,:));
    end
end

% Generate the psychometric curves for one of the models
[BIG_P_C_HAT] = load_P_C_HAT([1 1 1 1]); % load model predictions
[p_C_hat_mat,HR,FA] = compute_psych_curves(10,1,1,BIG_P_C_HAT); % plot curves

% Compute model log posterior
bars = zeros(size(model_mat,1),length(Subj_data_cell));
for model_num = 1
    bars(model_num,:) = compute_BMC(model_mat(model_num,:));
end

% Subtract log posterior of the best model from the rest, and plot results
bars_plot = (sum(bars,2) - max(sum(bars,2)));
bar(bars_plot);
set(gca,'XTickLabel',{'VVO','VEO','VAO','VSO','VVM','VEM','VAM','VSM',...
    'EEO','EAO','ESO','EEM','EAM','ESM'});
