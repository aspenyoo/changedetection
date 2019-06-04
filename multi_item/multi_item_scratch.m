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
% this is to continue going through the runlist, based on what runlists
% were complete already

clear all

% filename = 'analysis/clusterfittingsettings.mat';
filename = 'analysis/clusterfitting_joint.mat';
% filename = 'analysis/clusterjobs_keshvari.mat';
additionalpaths = ''; %'ellipse_keshvari/'

% subjidCell = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
subjidCell = {'S02','S03','S06','S07','S08','S10','S11','S14','S04'};
conditionCell = {'combined'};
modelMat = ...
    [1 1 1;  1 2 1; ...  % V_O model variants
     1 1 2;  1 2 2; ...  % V_M model variants
             2 2 1; ...  % F_O model variants
             2 2 2];     % F_M model variants
% modelMat = ...
%     [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
%      1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
%              2 2 1; 2 3 1; ...  % F_O model variants
%              2 2 2; 2 3 2];     % F_M model variants
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
            load(sprintf('analysis/fits/%ssubj%s_%s_model%d%d%d.mat',additionalpaths,subjid,condition,model(1),model(2),model(3)));
            
            % get the indices of runlists left
            incompleteRuns = 1:20;
            incompleteRuns(completedruns) = [];
            nRunsleft = length(incompleteRuns);
            
            % assume that the number completed is the amount that can be
            % completed in the same amount of time. set number of jobs
            % based on that
            nRunsperJob = 2;%length(completedruns)*2;
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

%% ======================================================================
%                       GETTING MODEL FITS
% ======================================================================
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

% subjidVec = {'S02','S03','S06','S08','S10','S11','S14'};
subjidVec = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari/'; % ellipse_keshvari
additionalmodifier = '_keshvari';

% modelMat = [1 1 1; 1 1 2; 1 3 1; 1 3 2];
modelMat = ...
    [1 1 1;  1 2 1;  1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2;  1 3 2; ...  % V_M model variants
             2 2 1;  2 3 1; ...  % F_O model variants
             2 2 2;  2 3 2];     % F_M model variants
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
            load(sprintf('analysis/fits/%ssubj%s_%s_model%d%d%d.mat',additionalpaths,subjid,condition,model(1),model(2),model(3)))
            
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

save(sprintf('analysis/fits/%sbfp_%s%s.mat',additionalpaths,condition,additionalmodifier),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

% save(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

%% model comparison (AICc and BIC)

clear all

condition = 'combined';
load(sprintf('analysis/fits/bfp_%s.mat',condition));
% foldername = 'ellipse_keshvari';
% load(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition));
% modelnames = {'VVO','VVM','VSO','VSM'};
modelnames = {  'VVO', 'VFO', 'VSO',...
                'VVM', 'VFM', 'VSM',...
                       'FFO', 'FSO',...
                       'FFM', 'FSM'};
nModels = length(modelnames);
nTrials = 2000;

% calculated AIC, AICc, and BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));

figure;
bar(AICcMat)
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot


figure;
bar(BICMat)
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot

%% AICc BIC relatve to VVO

AICcMat = bsxfun(@minus,AICcMat,AICcMat(2,:));
BICMat = bsxfun(@minus,BICMat,BICMat(2,:));

figure;
bar(AICcMat)
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot


figure;
bar(BICMat)
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot

%% mean sem of same thing
nSubj = 6; %size(AICcMat,2);

M_AICc = nanmean(AICcMat,2);
SEM_AICc = nanstd(AICcMat,[],2)/sqrt(nSubj);

M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

figure;
errorbar(M_AICc,SEM_AICc,'k','LineStyle','none')
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot

figure
errorbar(M_BIC,SEM_BIC,'k','LineStyle','none')
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:10,'XTickLabel',modelnames);
defaultplot



 
%% ====================================================================
%                      PLOTS: ONE CONDITION
% ====================================================================


%% DATA: all subjects, psychometric function

clear all
condition = 'Ellipse';

%  subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14'};
subjidVec = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
nSubj = length(subjidVec);

nBins = 8;
quantilebinedges = 11;
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

%% DATA: all subjects, hits false alarms

clear all
condition = 'Ellipse';

 subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14'};
% subjidVec = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
nSubj = length(subjidVec);


[HRallVec,HRlowVec,HRhighVec,FARVec] = deal(nan(nSubj,5));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};

    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

    % get hits false alarms
    [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:)] = ...
        plot_HR_FAR(data,[],0);
end

% get participant and model means
m_HRall = mean(HRallVec);
m_HRlow = mean(HRlowVec);
m_HRhigh = mean(HRhighVec);
m_FAR = mean(FARVec);
sem_HRall = std(HRallVec)./sqrt(nSubj);
sem_HRlow = std(HRlowVec)./sqrt(nSubj);
sem_HRhigh = std(HRhighVec)./sqrt(nSubj);
sem_FAR = std(FARVec)./sqrt(nSubj);


% plot
figure; hold on
errorbar(0:4,m_HRall,sem_HRall,'o-')
errorbar(0:4,m_HRlow,sem_HRlow,'o-')
errorbar(0:4,m_HRhigh,sem_HRhigh,'o-')
errorbar(0:4,m_FAR,sem_FAR,'o-')


%% SINGLE SUBJECT MODEL FITS: psychometric function

clear all
condition = 'Ellipse';
foldername = 'ellipse_keshvari';
subjidx = 2;
modelidx = 1;
nBins = 6;

subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14','POO','METEST'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);         
subjid = subjidVec{subjidx};

% % load bfp fits
load(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition))
bfp = bfpMat{modelidx}(subjidx,:);
bfp = [8 1 5 0.501];
% bfp = [22 7 21 0.5];
% bfp = [49.3333    0.4506    9.4590];

% load data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

% get predictions
nSamples = 500;
[LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);
LL

% plot it
figure;
quantilebinedges = 1;
plot_psychometric_fn(data,nBins,p_C_hat,quantilebinedges);


%%

nSamplesVec = [10 50 100 200 500];
nSamps = length(nSamplesVec);
for isamps = 1:nSamps
    tic;
    LL(isamps) = calculate_LL(bfp,data,model,[],nSamplesVec(isamps));
    timelength(isamps) = toc;
end

figure; 
subplot(1,2,1)
plot(nSamplesVec,LL)
subplot(1,2,2)
plot(nSamplesVec,timelength)


%% 
nSamplesVec = [25 50 100 200 500];
nSamps = length(nSamplesVec);
bfp = [8 1 5 0.501];

nn = 6;
% JbarlowVec = linspace(0.5,25,nn);
pchangeVec = linspace(0.1,0.9,nn)

for i = 1:nn
    i
%     bfp(1) = JbarlowVec(i);
bfp(4) = pchangeVec(i);
    
    for isamps = 1:nSamps
        tic;
        LL(i,isamps) = calculate_LL(bfp,data,model,[],nSamplesVec(isamps));
        timelength(i,isamps) = toc;
    end
    
end

figure; 
plot(JbarlowVec,LL)

%% single subject mode fits: hits false alarms

figure;
plot_HR_FAR(data,p_C_hat)


%% ALL SUBJ MODEL FITS: psychometric function and hits/false alarms

clear all
condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari/';
additionalpaths2 = '_keshvari';
modelidx = 3;

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);
         
% load bfp fits
load(sprintf('analysis/fits/%sbfp_%s%s.mat',additionalpaths,condition,additionalpaths2))
bfpMat = bfpMat{modelidx};
nSubj = length(subjidVec);

% prediction stuff
nSamples = 200;
nBins = 8;
quantilebinning=1;

figure;
[x_mean, pc_data, pc_pred] = deal(nan(5,nBins,nSubj));
[HRallVec,HRlowVec,HRhighVec,FARVec,mod_HRallVec,mod_HRlowVec,mod_HRhighVec,mod_FARVec] = deal(nan(nSubj,5));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};
    bfp = bfpMat(isubj,:);
    
    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

    % get predictions
    [LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);
    fprintf('subj %s: %5.2f \n',subjid,LL)

    % get psychometric function binned data
    [x_mean(:,:,isubj), pc_data(:,:,isubj), pc_pred(:,:,isubj)] = plot_psychometric_fn(data,nBins,p_C_hat,quantilebinning);
    
    % get hits/false alarms binned data
    [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:),...
        mod_HRallVec(isubj,:),mod_HRlowVec(isubj,:),mod_HRhighVec(isubj,:),mod_FARVec(isubj,:)] = ...
        plot_HR_FAR(data,p_C_hat,0);
    
end

% get participant and model means: psychometric fn
xrange = nanmean(x_mean,3);
partM = nanmean(pc_data,3);
partSEM = nanstd(pc_data,[],3)./sqrt(nSubj-1);
modelM = nanmean(pc_pred,3);
modelSEM = nanstd(pc_pred,[],3)./sqrt(nSubj-1);

% get participant and model means: hits/false alarms
m_HRall = mean(HRallVec);
m_HRlow = mean(HRlowVec);
m_HRhigh = mean(HRhighVec);
m_FAR = mean(FARVec);
sem_HRall = std(HRallVec)./sqrt(nSubj);
sem_HRlow = std(HRlowVec)./sqrt(nSubj);
sem_HRhigh = std(HRhighVec)./sqrt(nSubj);
sem_FAR = std(FARVec)./sqrt(nSubj);
m_mod_HRall = mean(mod_HRallVec);
m_mod_HRlow = mean(mod_HRlowVec);
m_mod_HRhigh = mean(mod_HRhighVec);
m_mod_FAR = mean(mod_FARVec);
sem_mod_HRall = std(mod_HRallVec)./sqrt(nSubj);
sem_mod_HRlow = std(mod_HRlowVec)./sqrt(nSubj);
sem_mod_HRhigh = std(mod_HRhighVec)./sqrt(nSubj);
sem_mod_FAR = std(mod_FARVec)./sqrt(nSubj);

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

% plot

figure;

subplot(1,2,1); hold on
for ii = 1:5;
plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
    modelSEM(ii,:),colorMat(ii,:),colorMat(ii,:))
end

subplot(1,2,2); hold on;
plot_summaryfit(0:4,m_HRall,sem_HRall,m_mod_HRall,sem_mod_HRall,colorMat(1,:),colorMat(1,:));
plot_summaryfit(0:4,m_HRlow,sem_HRlow,m_mod_HRlow,sem_mod_HRlow,colorMat(2,:),colorMat(2,:));
plot_summaryfit(0:4,m_HRhigh,sem_HRhigh,m_mod_HRhigh,sem_mod_HRhigh,colorMat(3,:),colorMat(3,:));
plot_summaryfit(0:4,m_FAR,sem_FAR,m_mod_FAR,sem_mod_FAR,colorMat(4,:),colorMat(4,:));




%% ====================================================================
%                      PLOTS: BOTH CONDITION
% ====================================================================

%% SINGLE SUBJECT MODEL FITS: combined

clear all
condition = 'combined';
% disptype = 'same';
subjidx = 4;
modelidx = 1;
nBins = 8;
nSamples = 200;

% get correct settings
subjidVec = {'S02','S03','S06','S08','S10','S11','S14'};
modelMat = [1 1 1; 1 1 2; 1 3 1; 1 3 2];

model = modelMat(modelidx,:);         
subjid = subjidVec{subjidx};
infering = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% load ML parameter estimates
load(sprintf('analysis/fits/bfp_%s.mat',condition))
% load(sprintf('analysis/fits/%sdisp/bfp_%s.mat',disptype,condition))
bfp = bfpMat{modelidx}(subjidx,:);

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

%% model fits of all subject

clear all
condition = 'combined';
% disptype = 'same';
imodel = 1;
nBins = 6;
nSamples = 200;

% get correct settings
subjidVec = {'S02','S03','S06','S08','S10','S11','S14'};
modelMat = [1 1 1; 1 1 2; 1 3 1; 1 3 2];
nSubj = length(subjidVec);
nModels = size(modelMat,1);

model = modelMat(imodel,:);         

% load ML parameter estimates
load(sprintf('analysis/fits/bfp_%s.mat',condition))
% load(sprintf('analysis/fits/%sdisp/bfp_%s.mat',disptype,condition))

% [p_C_hat_Line, p_C_hat_Ellipse] = deal(2000,nSubj);
% % get predictions for each subject
% for isubj = 1:nSubj
% [LL,pchat] = calculate_joint_LL(bfp,data_E,data_L,model,[],nSamples);
% LL
% p_C_hat_Line(:,isubj) = pchat.Line;
% p_C_hat_Ellipse(:,isubj) = pchat.Ellipse;
% end


[x_mean_e, pc_data_e, pc_pred_e, x_mean_l, pc_data_l, pc_pred_l] = deal(nan(5,nBins,nSubj));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};
    bfp = bfpMat{imodel}(isubj,:);
    
    % load data
    load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid),'data');
    data_E = data;
    load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid),'data');
    data_L = data;
    
    % get predictions
    [LL,p_C_hat] = calculate_joint_LL(bfp,data_E,data_L,model,[],nSamples);
    fprintf('subj %s: %5.2f \n',subjid,LL)

    [x_mean_e(:,:,isubj), pc_data_e(:,:,isubj), pc_pred_e(:,:,isubj)] = plot_psychometric_fn(data_E,nBins,p_C_hat.Ellipse,0);
    [x_mean_l(:,:,isubj), pc_data_l(:,:,isubj), pc_pred_l(:,:,isubj)] = plot_psychometric_fn(data_L,nBins,p_C_hat.Line,0);
end

% get participant and model means
xrange_e = nanmean(x_mean_e,3);
partM_e = nanmean(pc_data_e,3);
partSEM_e = nanstd(pc_data_e,[],3)./sqrt(nSubj-1);
modelM_e = nanmean(pc_pred_e,3);
modelSEM_e = nanstd(pc_pred_e,[],3)./sqrt(nSubj-1);

xrange_l = nanmean(x_mean_l,3);
partM_l = nanmean(pc_data_l,3);
partSEM_l = nanstd(pc_data_l,[],3)./sqrt(nSubj-1);
modelM_l = nanmean(pc_pred_l,3);
modelSEM_l = nanstd(pc_pred_l,[],3)./sqrt(nSubj-1);

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

figure;
subplot(1,2,1)
for ii = 1:5;
plot_summaryfit(xrange_e(ii,:),partM_e(ii,:),partSEM_e(ii,:),modelM_e(ii,:),...
    modelSEM_e(ii,:),colorMat(ii,:),colorMat(ii,:))
end
title('Ellipse')
defaultplot

subplot(1,2,2)
for ii = 1:5;
plot_summaryfit(xrange_l(ii,:),partM_l(ii,:),partSEM_l(ii,:),modelM_l(ii,:),...
    modelSEM_l(ii,:),colorMat(ii,:),colorMat(ii,:))
end
title('Line')
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
