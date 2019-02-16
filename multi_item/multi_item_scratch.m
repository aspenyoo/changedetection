
%% set up data into useable format

subj_ID_cell = {'POO','METEST','POO','METEST'};
pres2stimuli_cell = {'Ellipse','Ellipse','Line','Line'};
Subj_data_cell = combine_convert_all_data(subj_ID_cell,pres2stimuli_cell);

save('analysis/Subj_data_cell.mat','Subj_data_cell')

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

%% stats about difference in performance as a function of condition

clear all
subjid_cell = {'POO','METEST'};
pres2stimuli_cell = {'Line','Ellipse'};
N = 4;
highrel_vec = 0:N; % number of high reliability stimuli
PC_mat = nan(length(subjid_cell), length(pres2stimuli_cell), length(highrel_vec));

for isubj = 1:length(subjid_cell);
    subjid = subjid_cell{isubj};
    
    for icond = 1:length(pres2stimuli_cell)
        pres2stimuli = pres2stimuli_cell{icond};
        
        % load subject data
        load(sprintf('data/combined_data/%s_%s_combined.mat',subjid,pres2stimuli),'trialMat');
        
        % get data in correct form
        was_change = trialMat(:,1) > 0;
        respond_change = trialMat(:,2);
        num_highrel = sum(trialMat(:,39:(39+N-1))==0.9,2);
        
        for irel = 1:length(highrel_vec)
            rel = highrel_vec(irel);
            idx = num_highrel == rel;
            
            PC_mat(isubj,icond,irel) = mean(was_change(idx) == respond_change(idx));
        end
        
    end
end


%% save data in format for fitting data
clear all

subjid = 'METEST'; 
pres2stimuli = 'Ellipse';
load(sprintf('data/combined_data/%s_%s_combined.mat',subjid,pres2stimuli),'trialMat');

% ======== EDIT DATA FORMAT =============
% data = trialMat(1:200,:);
data = trialMat;

nItems = 4;
nTrials = size(data,1);

data_rel = data(:,39:(38+nItems));   % reliabilities of items on first presentation
data_Delta = -data(:,56:(55+nItems))*(pi/90) + data(:,64:(63+nItems))*(pi/90); % y-x
data_resp = data(:,2);             % subject response (change/no change)

% first, sort by item reliabilities within each trial
[data_rel_sorted, I_rel_sorted] = sort(data_rel,2);
data_Delta_sorted = nan(size(data_Delta));
for itrial = 1:nTrials
    data_Delta_sorted(itrial,:) = data_Delta(itrial,I_rel_sorted(itrial,:));
end

% next, sort across trials by number of high reliability items (ascending)
rels = unique(data_rel);            % unique reliabilities across experiment
high_num = sum(data_rel==rels(2),2);% number of high rel items on each trial
[~, I] = sort(high_num);  % sorted (ascending order) and original indices of high_num

datt.Delta = data_Delta_sorted(I,:);          % sorted trial delta
datt.rel = data_rel_sorted(I,:);       % sorted reliability
datt.resp = data_resp(I);                   % sorted subject response
datt.pres2stimuli = pres2stimuli;
datt.subjid = subjid;

data = datt;

save(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,pres2stimuli),'data')

%% 
% CALC LILEIHOOD

model = [1 2 1]; % encoding, inference, decision rule
theta = [2 1 0.3 0.5];
nSamples = 200;

tic;
[LL,pc] =calculate_LL(theta,datt,model,pres2stimuli,nSamples);
LL
toc
i
%% fit parameter one time

model = [2 2 2]; % encoding, inference, decision rule
nSamples = 200;

runlist = 1;
runmax = 1;
nSamples = 20;

[bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples)



%% fit models
clear all

subjid = 'POO';
pres2stimuli = 'Ellipse';

model_mat = ...
    [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ... % VPO model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ... % VPM model variants
    2 2 1 1;  2 3 1 1; 2 4 1 1; ...  % EPO model variants
    2 2 2 1;  2 3 2 1; 2 4 2 1]; % EPM model variants

for imodel = 1:size(model_mat,1)
    run_model_reliability(subjid, pres2stimuli, model_mat(imodel,:));
end


%% model comparson
clear all

subj_id_cell = {'METEST','POO'};
pres2stimuli_cell = {'Ellipse','Ellipse'};

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
    
    LLMat(:,imodel) = compute_BMC(currmodel,subj_id_cell,pres2stimuli_cell);
end

bar(bsxfun(@minus,LLMat,max(LLMat,[],2))')
