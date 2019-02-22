% ============================================================
%                   DATA RELATED
% =============================================================
% 
%% old -- keshvari. set up data into useable format

subj_ID_cell = {'POO','METEST','POO','METEST'};
conditionCell = {'Ellipse','Ellipse','Line','Line'};
Subj_data_cell = combine_convert_all_data(subj_ID_cell,conditionCell);

save('analysis/Subj_data_cell.mat','Subj_data_cell')

%% old -- keshvari. plot psychometric function of current dataset

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
subjidCell = {'POO','METEST'};
conditionCell = {'Line','Ellipse'};
N = 4;
highrel_vec = 0:N; % number of high reliability stimuli
PC_mat = nan(length(subjidCell), length(conditionCell), length(highrel_vec));

for isubj = 1:length(subjidCell);
    subjid = subjidCell{isubj};
    
    for icond = 1:length(conditionCell)
        pres2stimuli = conditionCell{icond};
        
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


%% ==============================================================
%           MODEL FITTING
%  ===============================================================

%% look at gamma

Jbar = 12;
Jbar2 = 13;
tau = 5;
tau2 = 7; 

samps = nan(1e3,2);
samps(:,1) = gamrnd(Jbar/tau, tau, 1e3, 1);
samps(:,2) = gamrnd(Jbar2/tau, tau, 1e3, 1);

hist(samps,100)

%% simulate fake data (for parameter and model recovery)

clear all
close all

model = [1 1 1];
modelidx = 1;
condition = 'Ellipse';

% load any subject's data (just to get the deltas and reliabilities)
load('data/fitting_data/POO_Ellipse_simple.mat')
data.subjid = 'FAKE03';
data.pres2stimuli = condition;

% get theta value (made up or from fits)
load(sprintf('analysis/fits/bfp_%s.mat',condition))
bfpMat = bfpMat{modelidx};
M = mean(bfpMat);
sem = std(bfpMat)./size(bfpMat,1);
% theta = [6 2 8 0.5];
theta = sem.*randn(1,size(bfpMat,2))+M

% generate p_C_hat
nSamples = 200;
[~,p_C_hat] = calculate_LL(theta,data,model,[],nSamples);

% generate fake data
data.resp = rand(length(p_C_hat),1) < p_C_hat;

% plot to make sure it makes sense
plot_psychometric_fn(data,6,p_C_hat);

% save
save(sprintf('data/fitting_data/%s_%s_simple.mat',data.subjid,condition),'theta','p_C_hat','data')

%% calc likelihood of single condition

clear all

model = [1 2 1]; % encoding, inference, decision rule
theta = [2 1 0.3 0.5];
nSamples = 200;

tic;
[LL,pc] =calculate_LL(theta,datt,model,pres2stimuli,nSamples);
LL
toc

%% calc likelihood of joint condition

clear all
subjid = 'POO';
nSamples = 200;
model = [1 1 1];
theta = [2 1 3 0.3 0.5];

[~,~,~,~,logflag] = getFittingSettings(model, 'Line');

% load data and save as appropriate variables
load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid))
data_E = data;
load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid))
data_L = data;

LL = calculate_joint_LL(theta,data_E,data_L,model,logflag,nSamples)

%% fit model one time
% actually going to be done on cluster. see fit_parameters.s

model = [2 2 2]; % encoding, inference, decision rule
nSamples = 200;

runlist = 1;
runmax = 1;
nSamples = 20;

[bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples)

%% make mat file of settings for different cluster indices

% clear all

filename = 'analysis/clusterfittingsettings.mat';

subjidCell = {'POO','METEST'};
conditionCell = {'Ellipse','Line'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
nSubj = length(subjidCell);
nConds = length(conditionCell);
nModels = size(modelMat,1);
runlistpermodel = ones(1,nModels);

counter = 1;
for isubj = 1:nSubj
    subjid = subjidCell{isubj};
    
    for icond = 1:nConds
        condition = conditionCell{icond};
        
        for imodel = 1:nModels
            model = modelMat(imodel,:);
            
            load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)));
            
            incompleteRuns = 1:50;
            incompleteRuns(completedruns) = [];
            nRunsleft = length(incompleteRuns);
            
            for irun = 1:nRunsleft
                runlist = incompleteRuns(irun);
                
                clustersettings{counter}.subjid = subjid;
                clustersettings{counter}.condition = condition;
                clustersettings{counter}.model = model;
                clustersettings{counter}.runlist = runlist;
            
            counter = counter+1;
            end
        end
    end
    
end

save(filename,'clustersettings')


%% load current fits of models

clear all

subjid = 'POO';
condition = 'Ellipse';
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
nModels = size(modelMat,1);

nCompleteVec = nan(1,nModels);
for imodel = 1:nModels
    imodel
    model = modelMat(imodel,:);
    
    load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)))
    disp(completedruns')
    nCompleteVec(imodel) = length(completedruns);
end

%% get best fit param for subjects, for a particular condition

clear all

subjidVec = {'POO','METEST'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
nSubj = length(subjidVec);    
nModels = size(modelMat,1);
condition = 'Line';
% 
% for isubj = 1:nSubj
%     subjid = subjidVec{isubj}
%     
%     load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))
%     
%     for imodel = 1:nModels;
%         model = modelMat(imodel,:)    
%    
%         load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)))
%         
%         nRuns = length(completedruns);
%         for irun = 1:nRuns;
%             LLVec(irun) = -calculate_LL(bfp(irun,:),data,model,[],200);
%         end
%         save(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)),...
%             'completedruns','LLVec','bfp')
%         
%     end
% end
% 
% %%

LLMat = nan(nModels,nSubj);
bfpMat = cell(1,nModels);
nParamsVec = nan(1,nModels);
for imodel = 1:nModels;
    model = modelMat(imodel,:);
    
    for isubj = 1:nSubj
        subjid = subjidVec{isubj};
        
        load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)))

        if (isubj==1); 
            nParamsVec(imodel) = size(bfp,2);
            bfpMat{imodel} = nan(nSubj, nParamsVec(imodel)); 
        end
        
        idx_posLL = LLVec >0;
        idx_minLL = find(LLVec==min(LLVec(idx_posLL)),1,'first');
        LLMat(imodel,isubj) = LLVec(idx_minLL);
        bfpMat{imodel}(isubj,:) = bfp(idx_minLL,:);
        
    end
    
end

save(sprintf('analysis/fits/bfp_%s.mat',condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

%% model comparison (AICc and BIC)

clear all

condition = 'Ellipse';
load(sprintf('analysis/fits/bfp_%s.mat',condition));
modelnames = {  'VVO', 'VFO', 'VSO',...
                'VVM', 'VFM', 'VSM',...
                       'FFO', 'FSO',...
                       'FFM', 'FSM'};

nTrials = 2000;

% calculated AIC, AICc, and BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));

figure;
bar(AICcMat)
title('AICc')
xlim([0.5 10.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot


figure;
bar(BICMat)
title('BIC')
xlim([0.5 10.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot

%% old fit models (keshvari way)

clear all

subjid = 'POO';
pres2stimuli = 'Ellipse';

modelMat = ...
    [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ... % VPO model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ... % VPM model variants
    2 2 1 1;  2 3 1 1; 2 4 1 1; ...  % EPO model variants
    2 2 2 1;  2 3 2 1; 2 4 2 1]; % EPM model variants

for imodel = 1:size(modelMat,1)
    run_model_reliability(subjid, pres2stimuli, modelMat(imodel,:));
end

%% old model comparison (keshvari way)

clear all

subjidCell = {'METEST','POO'};
conditionCell = {'Ellipse','Ellipse'};

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants

nModels = size(modelMat,1);
nSubj = length(subjidCell);

LLMat = nan(nSubj,nModels);
for imodel = 1:nModels
    currmodel = modelMat(imodel,:);
    
    LLMat(:,imodel) = compute_BMC(currmodel,subjidCell,conditionCell);
end

bar(bsxfun(@minus,LLMat,max(LLMat,[],2))')


%% ====================================================================
%                   PLOTS
% ====================================================================

%% model fits

clear all
condition = 'Line';
subjidx = 1;
modelidx = 10;
nBins = 6;

subjVec = {'POO','METEST'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);         
subjid = subjVec{subjidx};

% load bfp fits
load(sprintf('analysis/fits/bfp_%s.mat',condition))
bfp = bfpMat{modelidx}(subjidx,:);

% load data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

% get predictions
nSamples = 200;
[LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);

% plot it
LL
figure;
plot_psychometric_fn(data,nBins,p_C_hat)

