%% ============================================================
%                   DATA RELATED
% =============================================================
% 

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

subjid = 'S14'; 
pres2stimuli = 'Line';

trialMat = combine_data(subjid, pres2stimuli);

% load(sprintf('data/combined_data/%s_%s_combined.mat',subjid,pres2stimuli),'trialMat');

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
%               QUALITATIVE DIFFERENCES IN DATA
%  ===============================================================

clear all

subjidVec = {'POO','METEST'};
nSubj = length(subjidVec);

conditionVec = {'Ellipse','Line'};
nConditions = length(conditionVec);

PC = nan(nSubj,5,nConditions);
for icondition = 1:nConditions
    condition = conditionVec{icondition};
    
    for isubj = 1:nSubj;
        subjid = subjidVec{isubj};
        
        % load data
        load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))
        data_waschange = sum(data.Delta,2)>0;
        data_nrels = sum(data.rel == 0.9,2);
        data_correct = data.resp == data_waschange;
        
        for irel = 0:4;
            idx = data_nrels==irel;
            PC(isubj,irel+1,icondition) = mean(data_correct(idx));
            
        end
        
    end
end

figure; hold on
boxplot(PC(:,:,1),'Colors','r')
boxplot(PC(:,:,2),'Colors','b')
title('line')
xlabel('nrels')
ylabel('PC')
ylim([0.5 0.8])
defaultplot

%% ==============================================================
%                     MODEL FITTING
%  ===============================================================

%% look at gamma

Jbar = 24;
Jbar2 = 24;
tau = 1;
tau2 = 3; 

samps = nan(1e3,2);
samps(:,1) = gamrnd(Jbar/tau, tau, 1e3, 1);
samps(:,2) = gamrnd(Jbar2/tau2, tau2, 1e3, 1);

mean(samps)
hist(samps,100)


%% calc likelihood of single condition

clear all
subjid = 'POO';
condition = 'Ellipse';

model = [2 2 3]; % encoding, inference, decision rule
theta = [2 1 2];
nSamples = 200;

load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))

tic;
[LL,pc] =calculate_LL(theta,data,model,[],nSamples);
LL
toc

%% calc likelihood of joint condition

% clear all
subjid = 'POO';
nSamples = 200;
model = [1 1 1];
% theta = [1.56303725054700,0.453115357333784,2.73752936332438,2.00893833208493,0.743587962404854];
theta = [13.4925771287636,25.4226227004200,5.86566220663474,9.64037313579704,0.551541656814516];

logflag = getFittingSettings(model, 'Line');
theta(logflag) = log(theta(logflag));

% load data and save as appropriate variables
load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid))
data_E = data;
load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid))
data_L = data;

tic;
LL = calculate_joint_LL(theta,data_E,data_L,model,logflag,nSamples)
toc
%% fit model one time
% actually going to be done on cluster. see fit_parameters.s

clear all
subjid = 'POO';

load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid))

model = [2 2 2]; % encoding, inference, decision rule
nSamples = 200;
runlist = 1;
runmax = 1;

[bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples)

%% make mat file of settings for different cluster indices

clear all

filename = 'analysis/clusterfittingsettings.mat';

subjidCell = {'POO','METEST','S02','S04','S06','S08'};
conditionCell = {'Ellipse'};
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
            try
            load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,condition,model(1),model(2),model(3)));
            
            % get the indices of runlists left
            incompleteRuns = 1:20;
            incompleteRuns(completedruns) = [];
            nRunsleft = length(incompleteRuns);
            
            % assume that the number completed is the amount that can be
            % completed in the same amount of time. set number of jobs
            % based on that
            nRunsperJob = length(completedruns)*2;
            while (~isempty(incompleteRuns)) % while there are runs not assigned to jobs
                clustersettings{counter}.subjid = subjid;
                clustersettings{counter}.condition = condition;
                clustersettings{counter}.model = model;
                
                try
                    clustersettings{counter}.runlist = incompleteRuns(1:nRunsperJob);
                    incompleteRuns(1:nRunsperJob) = [];
                catch
                    clustersettings{counter}.runlist = incompleteRuns;
                    incompleteRuns = [];
                end
                
                counter = counter+1;
            end
            end
%             for irun = 1:nRunsleft
%                 runlist = incompleteRuns(irun);
%                 
%                 clustersettings{counter}.subjid = subjid;
%                 clustersettings{counter}.condition = condition;
%                 clustersettings{counter}.model = model;
%                 clustersettings{counter}.runlist = runlist;
%             
%             counter = counter+1;
%             end
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

subjidVec = {'POO','METEST','S02','S03','S06','S08','S10','S11','S14'};
% subjidVec = {'POO'};
condition = 'Ellipse';
foldername = 'ellipse_keshvari';


modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
nSubj = length(subjidVec);    
nModels = size(modelMat,1);


LLMat = nan(nModels,nSubj);
bfpMat = cell(1,nModels);
nParamsVec = nan(1,nModels);
for imodel = 1:nModels;
    model = modelMat(imodel,:);
    
    for isubj = 1:nSubj
        subjid = subjidVec{isubj};
        
        try
            load(sprintf('analysis/fits/%s/subj%s_%s_model%d%d%d.mat',foldername,subjid,condition,model(1),model(2),model(3)))
            
            if (isubj==1);
                nParamsVec(imodel) = size(bfp,2);
                bfpMat{imodel} = nan(nSubj, nParamsVec(imodel));
            end
            
            idx_posLL = LLVec >0;
            idx_minLL = find(LLVec==min(LLVec(idx_posLL)),1,'first');
            LLMat(imodel,isubj) = LLVec(idx_minLL);
            bfpMat{imodel}(isubj,:) = bfp(idx_minLL,:);
        catch
            fprintf('model %d%d%d does not exist for subject %s \n',...
                model(1),model(2),model(3),subjid)
        end
    end
    
end

save(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

%% model comparison (AICc and BIC)

clear all

condition = 'Ellipse';
foldername = 'ellipse_keshvari';
load(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition));
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

%% AICc BIC relatve to VVO

AICcMat = bsxfun(@minus,AICcMat,AICcMat(1,:));
BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

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

%% mean sem of same thing
nSubj = 6;%size(AICcMat,2);

M_AICc = nanmean(AICcMat,2);
SEM_AICc = nanstd(AICcMat,[],2)/sqrt(nSubj);

M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

figure;
errorbar(M_AICc,SEM_AICc,'k','LineStyle','none')
title('AICc')
xlim([0.5 10.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot

figure
errorbar(M_BIC,SEM_BIC,'k','LineStyle','none')
title('BIC')
xlim([0.5 10.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot


%% ====================================================================
%               PARAMETER/MODEL RECOVERY
% =====================================================================


%% simulate fake data (for parameter and model recovery)

clear all
% close all

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
imodel = 1;
model = modelMat(imodel,:);
condition = 'Ellipse';

% load any subject's data (just to get the deltas and reliabilities)
load('data/fitting_data/POO_Ellipse_simple.mat')
data.subjid = sprintf('F_%d%d%d_01',model(1),model(2),model(3));
data.pres2stimuli = condition;

% % get theta value (made up or from fits)
% load(sprintf('analysis/fits/bfp_%s.mat',condition))
% bfpMat = bfpMat{imodel};
% M = mean(bfpMat);
% sem = std(bfpMat)./size(bfpMat,1);

theta =  [24 12 2 0.5];
% theta = sem.*randn(1,size(bfpMat,2))+M

% generate p_C_hat
nSamples = 200;
[~,p_C_hat] = calculate_LL(theta,data,model,[],nSamples);

% generate fake data
data.resp = rand(length(p_C_hat),1) < p_C_hat;

% plot to make sure it makes sense
clf
plot_psychometric_fn(data,6,p_C_hat);

% save
% save(sprintf('data/fitting_data/%s_%s_simple.mat',data.subjid,condition),'theta','p_C_hat','data')

%% load actual and estimated parameter
clear all

subjid = 'FAKE01';
condition = 'Ellipse';
truemodel = [1 1 1];
estmodel = truemodel;

% load data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))

% load fits
load(sprintf('subj%s_%s_model%d%d%d.mat',subjid,condition,estmodel(1),estmodel(2),estmodel(3)))

% print stuff
theta
bfp

% find best fitting parameters
idx_minLL = find(LLVec==min(LLVec),1,'first');
BFP = bfp(idx_minLL,:)

%% calculate KL divergence
% (run cell above before this one)

% get predictions of best fit param
[~,q_C_hat] = calculate_LL(BFP,data,estmodel);

p_logp = sum(data.resp.*p_C_hat.*log(p_C_hat)) + sum((1-data.resp).*(1-p_C_hat).*log(1-p_C_hat))
p_logq = sum(data.resp.*p_C_hat.*log(q_C_hat)) + sum((1-data.resp).*(1-p_C_hat).*log(1-q_C_hat))
- p_logp - p_logq
 
%% look at LL for different models

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
nModels = size(modelMat,1);

LL_vec = nan(1,nModels);
for imodel = 1:nModels
    estmodel = modelMat(imodel,:);
    
    load(sprintf('subj%s_%s_model%d%d%d.mat',subjid,condition,estmodel(1),estmodel(2),estmodel(3)))

    idx_LL = find(LLVec == min(LLVec),1,'first');
    LL_vec(imodel) = LLVec(idx_LL);
    
end

LL_vec

 
%% ====================================================================
%                         PLOTS
% ====================================================================


%% DATA: all subjects for Ellipse/Line

clear all
condition = 'Line';

subjidVec = {'POO','METEST','S02','S03','S06','S07','S08','S10','S11','S14'};
nSubj = length(subjidVec);

nBins = 6;
quantilebinedges = 0;
figure; 
[x_mean, pc_data] = deal(nan(5,nBins,nSubj));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};

    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

    [x_mean(:,:,isubj), pc_data(:,:,isubj)] = plot_psychometric_fn(data,nBins,[],quantilebinedges);
end

% get participant and model means
xrange = mean(x_mean,3);
partM = mean(pc_data,3);
partSEM = std(pc_data,[],3)./sqrt(nSubj);

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

clf
for ii = 1:5;
plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),[],...
    [],colorMat(ii,:),[])
end


%% SINGLE SUBJECT MODEL FITS: Ellipse/Line

clear all
condition = 'Ellipse';
foldername = 'ellipse_keshvari';
subjidx = 3;
modelidx = 1;
nBins = 6;

subjVec = {'POO','METEST','S02','S03','S04','S05','S06','S07','S08'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);         
subjid = subjVec{subjidx};

% % load bfp fits
load(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition))
bfp = bfpMat{modelidx}(subjidx,:);
% bfp = [49.3333    0.4506    9.4590];

% load data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

% get predictions
nSamples = 200;
[LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);
LL

% plot it
figure;
quantilebinedges = 0;
plot_psychometric_fn(data,nBins,p_C_hat,quantilebinedges);

%% SINGLE SUBJECT MODEL FITS: combined

clear all
condition = 'combined';
disptype = 'same';
subjidx = 1;
modelidx = 1;
nBins = 6;
nSamples = 200;

% get correct settings
subjVec = {'POO','METEST','S02','S03','S04','S05','S06','S07','S08'};
modelMat = ...
    [1 1 1;  1 2 1;  1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2;  1 3 2; ...  % V_M model variants
             2 2 1;  2 3 1; ...  % F_O model variants
             2 2 2;  2 3 2];     % F_M model variants
model = modelMat(modelidx,:);         
subjid = subjVec{subjidx};
infering = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% load ML parameter estimates
load(sprintf('analysis/fits/%sdisp/bfp_%s.mat',disptype,condition))
bfp = bfpMat{modelidx}(subjidx,:);

x_ellipse = bfp;
if (decision_rule == 1) && (infering == 3);
    idx_Lonly = [3 6];
else
    idx_Lonly = [3];
end
x_ellipse(idx_Lonly) = [];

% load data
load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid),'data');
data_E = data;
load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid),'data');
data_L = data;

% get predictions
[LL,p_C_hat] = calculate_joint_LL(bfp,data_E,data_L,model,[],nSamples);
LL

% plot it
figure;
quantilebinedges = 0;
subplot(1,2,1)
plot_psychometric_fn(data_E,nBins,p_C_hat.Ellipse,quantilebinedges);
subplot(1,2,2)
plot_psychometric_fn(data_L,nBins,p_C_hat.Line,quantilebinedges);


%% ALL SUBJ MODEL FITS: Ellipse/Line

clear all
condition = 'Ellipse';
modelidx = 1;

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);
         
% load bfp fits
load(sprintf('analysis/fits/bfp_%s.mat',condition))
bfpMat = bfpMat{modelidx};
nSubj = length(subjidVec);

% prediction stuff
nSamples = 200;
nBins = 6;

[x_mean, pc_data, pc_pred] = deal(nan(5,nBins,nSubj));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};
    bfp = bfpMat(isubj,:);
    
    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

    % get predictions
    [LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);
    fprintf('subj %s: %5.2f \n',subjid,LL)

    figure;
    [x_mean(:,:,isubj), pc_data(:,:,isubj), pc_pred(:,:,isubj)] = plot_psychometric_fn(data,nBins,p_C_hat,0);
    pause;
end

% get participant and model means
xrange = nanmean(x_mean,3);
partM = nanmean(pc_data,3);
partSEM = nanstd(pc_data,[],3)./sqrt(nSubj-1);
modelM = nanmean(pc_pred,3);
modelSEM = nanstd(pc_pred,[],3)./sqrt(nSubj-1);

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

figure;
for ii = 1:5;
plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
    modelSEM(ii,:),colorMat(ii,:),colorMat(ii,:))
end


%% plot of joint model fits

clear all

subjid = 'POO';
modelidx = 1;
nBins = 6;

conditionCell = {'Ellipse','Line'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);         

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
