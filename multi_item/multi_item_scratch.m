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

subjid = 'S23';
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
nSubjs = length(subjidVec);

conditionVec = {'Ellipse','Line'};
nConditions = length(conditionVec);

PC = nan(nSubjs,5,nConditions);
for icondition = 1:nConditions
    condition = conditionVec{icondition};
    
    for isubj = 1:nSubjs;
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

Jbar = 6;
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
subjid = 'S19';
condition = 'Ellipse';

model = [1 2 1]; % encoding, inference, decision rule
imodel = 2;

load('bfp_Ellipse.mat')
theta = bfpMat{imodel}(1,:);

% theta = [2 1 2];
nSamples = 50;

load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))

nCalc = 100;
for icalc = 1:nCalc
    icalc
    LL(icalc) =calculate_LL(theta,data,model,[],nSamples);
end

figure
histogram(LL)

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
nSubjs = length(subjidCell);
nConds = length(conditionCell);
nModels = size(modelMat,1);
runlistpermodel = ones(1,nModels);

counter = 1;
for isubj = 1:nSubjs
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

%% fit decision noise parameter based on the no decision noise minimum

clear all

icond = 2;
imodel = 8; % the og one without decision noise 
decision_noise = 1;     % that decision noise number
xxx = 21;

load('modelfittingsettings.mat')
options.UncertaintyHandling = 'on';
condition = conditionVec{icond};
model_og = modelMat(imodel,:);
model = model_og;
model(end) = decision_noise;
nSamples = [500 1000];

% get original bfp, in correct log space
load(sprintf('fits/%s/bfp_%s.mat',condition,condition))
logflag = getFittingSettings(model_og, condition);
x0_list = bfpMat{imodel};
x0_list(:,logflag) = log(x0_list(:,logflag));

% update logflag and other fitting parameters
[logflag,LB,UB,PLB,PUB] = getFittingSettings(model, condition);

for isubj = 1:nSubjs
    subjid = subjidVec{isubj};
    
    % load data
    load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))
    
    filename = sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,data.subjid,condition,model(1),model(2),model(3),model(4));
    filename
    
    % Fix random seed based on iteration (for reproducibility)
    rng(xxx);
    x0 = [x0_list(isubj,1:end-1) PLB(end-1)+rand*(PUB(end-1)-PLB(end-1)) x0_list(isubj,end)]
    x0(4) = x0(4) + 1e-3;
    [xbest,~,~,~] = ...
        bads(@(x) -calculate_LL(x,data,model,logflag,nSamples(1)),x0,LB,UB,PLB,PUB,[],options)
    
    % recalculate LL with more samples
    LL = -calculate_LL(xbest,data,model,logflag,nSamples(2));
    
    xbest(logflag) = exp(xbest(logflag)); % getting parameters back into natural units
    
    % it is necessary to reload the file at every iteration in case multiple processors are
    % saving the file at the same time
    if exist(filename,'file')
        load(filename,'bfp','LLVec','completedruns')
    else
        [bfp, LLVec, completedruns] = deal([]);
    end
    
    % update and save variables
    bfp = [bfp; xbest];
    LLVec = [LLVec; LL];
    completedruns = [completedruns; xxx];
    save(filename,'bfp','LLVec','completedruns')
    
end

%% check that decision noise model has higher LL than without dec noise

clear all

isubj = 1;
icond = 2;
imodel = 42; % the og one without decision noise 
decision_noise = 1;     % that decision noise number
nSamples = 1000;
xx = 1;

load('modelfittingsettings.mat')
subjid = subjidVec{isubj};
condition = conditionVec{icond};
model_og = modelMat(imodel,:)

% load bfp mat and data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition))
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition))

% calc og LL
x0 = bfpMat{imodel}(isubj,:);
rng(xx);
LL = -calculate_LL(x0,data,model_og,[],nSamples)

%%
% calculate w decision noise
model = model_og;
model(end) = decision_noise;
x0 = [x0(1:end-1) 0 x0(end)];
rng(xx);
LL = -calculate_LL(x0,data,model,[],nSamples)


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

%% get best fit param for all subjects, for a particular condition

clear all

icond = 1;

load('modelfittingsettings.mat')
condition = conditionVec{icond};
modelMat = modelMat(29:42,:);
modelnamesVec = modelnamesVec(29:42);

LLMat = nan(nModels,nSubjs);
bfpMat = cell(1,nModels);
nParamsVec = nan(1,nModels);
for imodel = 1:nModels;
    model = modelMat(imodel,:);
    
    for isubj = 1:nSubjs
        subjid = subjidVec{isubj};
        
        try
            load(sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,subjid,condition,model(1),model(2),model(3),model(4)))
            
            if (isubj==1);
                nParamsVec(imodel) = size(bfp,2);
                bfpMat{imodel} = nan(nSubjs, nParamsVec(imodel));
            end
            
            idx_posLL = LLVec >0;
            idx_minLL = find(LLVec==min(LLVec(idx_posLL)),1,'first');
            LLMat(imodel,isubj) = LLVec(idx_minLL);
            bfpMat{imodel}(isubj,:) = bfp(idx_minLL,:);
        catch
            fprintf('model %d%d%d%d does not exist for subject %s \n',...
                model(1),model(2),model(3),model(4),subjid)
        end
    end
    
end

save(sprintf('fits/%s/bfp_%s%s.mat',condition,condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

% save(sprintf('analysis/fits/%s/bfp_%s.mat',foldername,condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

%% =========
% multiple LL calculations
% ==============

%% redo LL calculations
% using either fixed sampling or ibs

clear all

icond = 1;
imodel = 31;
samplingtype = 'ibs';

load('modelfittingsettings.mat')
condition = conditionVec{icond};
model = modelMat(imodel,:);

nSamps = 10;    % samples of LL calc
logflag = [];

load(sprintf('fits/%s/bfp_%s%s.mat',condition,condition),'LLMat','bfpMat','subjidVec','modelMat','nParamsVec')

switch samplingtype
    case 'fixed'
        % fixed sampling settings
        nSamples = 1000; % samples in one LL calc
    case 'ibs'
        % ibs settings
        options_ibs = ibslike('defaults');
        options_ibs.Vectorized = 'on';
        options_ibs.Nreps = 10;
end

try
    switch samplingtype
        case 'fixed'
            load(sprintf('fits/%s/redo_LL_fixedsampling%d_model%d%d%d%d.mat',condition,nSamples,model(1),model(2),model(3),model(4)))
        case 'ibs'
            load(sprintf('fits/%s/redo_LL%d_ibs_model%d%d%d%d.mat',condition,options_ibs.Nreps,model(1),model(2),model(3),model(4)))
    end
    subjstart = find(sum(isnan(LL),2)==nSamps,1,'first');
    
catch
    [evaltime,LL] = deal(nan(nSubjs,nSamps));
    subjstart = 1;
end


for isubj = subjstart:nSubjs
    subjid = subjidVec{isubj}
    
    % load subj data
    load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))
    
    if strcmp(samplingtype,'ibs')
        % ibs settings
        dMat = data.Delta;
        rels = unique(data.rel);
        blah = data.rel;
        for irel = 1:length(rels)
            blah(blah == rels(irel)) = irel;
        end
        dMat = [dMat blah];
    end
    
    x = bfpMat{imodel}(isubj,:);
    
    for isamp = 1:nSamps
        
        tic;
        switch samplingtype
            case 'fixed'
                % calculating LL using fixed sampling
                LL(isubj,isamp) = calculate_LL(x,data,model,logflag,nSamples);
            case 'ibs'
                % calculating LL using ibs
                fun = @(xx,y) fun_LL(xx,y,model,condition,logflag,data.resp);
                LL(isubj,isamp) = ibslike(fun,x,data.resp,dMat,options_ibs);
        end
        evaltime(isubj,isamp) = toc;
    end
    
    range(LL,2)
    switch samplingtype
        case 'fixed'
            save(sprintf('fits/%s/redo_LL_fixedsampling%d_model%d%d%d%d.mat',condition,nSamples,model(1),model(2),model(3),model(4)),'LL','evaltime','nSamps','nSamples','subjidVec','model')
        case 'ibs'
            save(sprintf('fits/%s/redo_LL%d_ibs_model%d%d%d%d.mat',condition,options_ibs.Nreps,model(1),model(2),model(3),model(4)),'LL','evaltime','nSamps','subjidVec','options_ibs','model')
    end
end

%% histogram


isubj = 1
histogram(LL(isubj,:),10)


%% histogram of variances across noise and no noise models

clear all
load('fits/redo_LL.mat')

nModels = 42
varr = nan(1,nModels);
for imodel = 1:nModels
    blah = var(LL{imodel},0,2);
    varr(imodel) = median(blah);
end

figure; hold on
histogram(varr(1:14),10)
histogram(varr(15:28),10)
histogram(varr(29:42),10)

%% average across subjects for
clear all

condition = 'Line';
load(sprintf('fits/%s/redo_LL.mat',condition))

blah = cellfun(@(x) mean(x,2),LL,'UniformOutput',false);
blah = cell2mat(blah);

xx = 29:42;%15:28;%
bleh = blah(:,xx)-blah(:,1:14);

figure
histogram(bleh(:),70)
defaultplot
xlabel('complex model better')

%% make hsitograms of the diff models with diff decision rule
imodel = 1;
isubj = 1;

histogram(LL{imodel}(isubj,:))
hold on; pause
histogram(LL{imodel+14}(isubj,:))


%% cluster fix
% i started fitting using ibs without moving previous fixed sample ones
% away. now separating the existing fits

clear all

condition = 'Ellipse';
subjVec = {'S02', 'S03','S06','S08','S10','S11','S14','S15',...
    'S16','S17','S19','S20','S23'};
modelMat = [1 1 1 2;    1 2 1 2;    1 3 1 2;    1 4 1 2; ...
            1 1 2 2;    1 2 2 2;    1 3 2 2;    1 4 2 2; ...
                        2 2 1 2;    2 3 1 2;    2 4 1 2; ...
                        2 2 2 2;    2 3 2 2;    2 4 2 2];
                    
nModels = size(modelMat,1);
nSubjs = length(subjVec);

for isubj = 1:nSubjs
    subjid = subjVec{isubj};
    
    for imodel = 1:nModels;
        model = modelMat(imodel,:);
        
        load(sprintf('subj%s_%s_model%d%d%d%d.mat',subjid,condition,model(1),...
            model(2),model(3),model(4)));
        
        idx = find(diff(completedruns)~=1);
        
        if length(idx)~=1;
            subjid;
            model;
            idx;
        else
            % rename vars so can overwrite
            bfp_og = bfp;
            completedruns_og = completedruns;
            LLVec_og = LLVec; 
            
            % save older, fixed sampling stuff
            bfp = bfp_og(1:idx,:);
            completedruns = completedruns_og(1:idx);
            LLVec = LLVec_og(1:idx);
            save(sprintf('fits/%s/zz_nSamples500/subj%s_%s_model%d%d%d%d.mat',condition,subjid,condition,model(1),...
                model(2),model(3),model(4)),'bfp','completedruns','LLVec')
            
            % save new, ibs stuff
            bfp = bfp_og(idx+1:end,:);
            completedruns = completedruns_og(idx+1:end);
            LLVec = LLVec_og(idx+1:end);
            save(sprintf('fits/%s/blah/subj%s_%s_model%d%d%d%d.mat',condition,subjid,condition,model(1),...
                model(2),model(3),model(4)),'bfp','completedruns','LLVec')
            
            % delete og file
            delete(sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,subjid,condition,model(1),...
                model(2),model(3),model(4)))
        end
            
    end

end


%% get cluster settings

clear all

% all possible things
runlist_og = 1:20;
subjidVec = {'S02','S03','S06','S08','S10','S11','S14','S15','S16','S17','S19','S20','S23'}; % all real full subjects
modelMat = ...
   [1 1 1 2;  1 2 1 2; 1 3 1 2; 1 4 1 2; ...  % V_O model variants --|
    1 1 2 2;  1 2 2 2; 1 3 2 2; 1 4 2 2; ...  % V_M model variants   |  global decision
              2 2 1 2; 2 3 1 2; 2 4 1 2; ...  % F_O model variants   |      noise
              2 2 2 2; 2 3 2 2; 2 4 2 2];     % F_M model variants __|
conditionVec = {'Ellipse','Line'};

nSubjs = length(subjidVec);
nModels = size(modelMat,1);
nConds = length(conditionVec);

% see what remains
condition = 'Ellipse';
[subjidCell, modelCell, conditionCell, runlistCell] = deal([]);
for isubj= 1:nSubjs
    subjid = subjidVec{isubj};
    
    for imodel = 1:nModels;
        model = modelMat(imodel,:);
        
        load(sprintf('subj%s_%s_model%d%d%d%d.mat',subjid,condition,model(1),...
            model(2),model(3),model(4)));
        
        remainingrunlist = runlist_og;
        remainingrunlist(completedruns) = [];
        
        for idx = remainingrunlist
            subjidCell = [subjidCell {subjid}];
            modelCell = [modelCell {model}];
            conditionCell = [conditionCell {condition}];
            runlistCell = [runlistCell {idx}];
        end
        
    end
end


save('jobsettings.mat','subjidCell','modelCell','conditionCell','runlistCell')


%% =======
%  model comparison
% ========

%% checking that noise models have higher LL than no noise models

clear all

icond = 1;

load('modelfittingsettings.mat')

condition = conditionVec{icond};
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition));

x = nModels/3;
blah = nan(x,nSubjs,3);
blah(:,:,1) = LLMat(1:14,:);
blah(:,:,2) = LLMat(15:28,:);
blah(:,:,3) = LLMat(29:42,:);
LL2 = bsxfun(@minus,blah(:,:,1),blah); 
% larger number indicates better fit for decieion noise model


figure;
for isub = 1:3
    subplot(3,1,isub)
    xx = [1:x]+(isub-1)*x;
    bar(LL2(:,:,isub))
    xlim([0.5 x+0.5])
    set(gca,'XTick',1:x,'XTickLabel',modelnamesVec(xx));
defaultplot
end

%% model comparison (AICc and BIC)

clear all

icond = 1;

load('modelfittingsettings.mat')

condition = conditionVec{icond};
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition));
nTrials = 2000;

% calculated AIC, AICc, and BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));

figure;
bar(AICcMat)
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot

figure;
bar(BICMat)
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot

%% AICc BIC relatve to VVO

AICcMat = bsxfun(@minus,AICcMat,AICcMat(1,:));
BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

figure;
bar(AICcMat)
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot


figure;
bar(BICMat)
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot


%% mean sem of same thing
nSubjs = size(AICcMat,2);

M_AICc = nanmean(AICcMat,2);
SEM_AICc = nanstd(AICcMat,[],2)/sqrt(nSubjs);

M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubjs);

figure;
bar(M_AICc); hold on
errorbar(M_AICc,SEM_AICc,'k','LineStyle','none')
title('AICc')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot

figure
bar(M_BIC); hold on
errorbar(M_BIC,SEM_BIC,'k','LineStyle','none')
title('BIC')
xlim([0.5 nModels+0.5])
set(gca,'XTick',1:nModels,'XTickLabel',modelnamesVec);
defaultplot

%% 3 separate plots, based on decision noise

x = 14;
figure;
for isub = 1:3
    subplot(3,1,isub)
    xx = [1:x]+(isub-1)*x;
    bar(BICMat(xx,:))
    xlim([0.5 x+0.5])
    set(gca,'XTick',1:x,'XTickLabel',modelnamesVec(xx));
defaultplot
end

%%
xx=0.45;

hold on
for imodel = 1:nModels
    m = M_BIC(imodel);
    sigma = SEM_BIC(imodel);
    fill(imodel+[-xx xx xx -xx -xx],m+[-sigma -sigma sigma sigma -sigma],0.7*ones(1,3))
end




%% ====================================================================
%                      PLOTS: ONE CONDITION
% ====================================================================

%% SINGLE SUBJECT MODEL FITS: psychometric function

clear all
condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari';
additionalpaths2 = '';%'_keshvari';
subjidx = 1;
modelidx = 1;
nBins = 6;

%subjidVec = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14','POO','METEST'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
    1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
    2 2 1; 2 3 1; ...  % F_O model variants
    2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);
subjid = subjidVec{subjidx};

% % load bfp fits
load(sprintf('analysis/fits/%s/bfp_%s%s.mat',additionalpaths,condition,additionalpaths2))
bfp = bfpMat{modelidx}(subjidx,:);
% bfp = [3.0309    1.3764    0.1887    0.3951];
% bfp = [22 7 21 0.5];
% bfp = [49.3333    0.4506    9.4590];

% load data
load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')

% get predictions
nSamples = 50;
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

%% model fits of all subjects

clear all
condition = 'combined';
additionalpaths = '';%'combined_diffdisp/';
% disptype = 'same';
imodel = 1;
nBins = 6;
nSamples = 200;

% get correct settings
subjidVec = {'S02','S03','S06','S08','S10','S11','S14',...
    'S15','S16','S17','S19','S20','S23'};
modelMat = [1 1 1; 1 1 2; 1 3 1; 1 3 2];
nSubjs = length(subjidVec);
nModels = size(modelMat,1);

model = modelMat(imodel,:);

% load ML parameter estimates
load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition))
% load(sprintf('analysis/fits/%sdisp/bfp_%s.mat',disptype,condition))

% [p_C_hat_Line, p_C_hat_Ellipse] = deal(2000,nSubj);
% % get predictions for each subject
% for isubj = 1:nSubj
% [LL,pchat] = calculate_joint_LL(bfp,data_E,data_L,model,[],nSamples);
% LL
% p_C_hat_Line(:,isubj) = pchat.Line;
% p_C_hat_Ellipse(:,isubj) = pchat.Ellipse;
% end

figure;
[x_mean_e, pc_data_e, pc_pred_e, x_mean_l, pc_data_l, pc_pred_l] = deal(nan(5,nBins,nSubjs));
for isubj = 1:nSubjs
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
partSEM_e = nanstd(pc_data_e,[],3)./sqrt(nSubjs-1);
modelM_e = nanmean(pc_pred_e,3);
modelSEM_e = nanstd(pc_pred_e,[],3)./sqrt(nSubjs-1);

xrange_l = nanmean(x_mean_l,3);
partM_l = nanmean(pc_data_l,3);
partSEM_l = nanstd(pc_data_l,[],3)./sqrt(nSubjs-1);
modelM_l = nanmean(pc_pred_l,3);
modelSEM_l = nanstd(pc_pred_l,[],3)./sqrt(nSubjs-1);

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
imodel = 10;
model = modelMat(imodel,:);
condition = 'Ellipse';

% load any subject's data (just to get the deltas and reliabilities)
load('data/fitting_data/POO_Ellipse_simple.mat')
data.pres2stimuli = condition;

% get theta value (made up or from fits)
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition))
bfpMat = bfpMat{imodel};
m = mean(bfpMat);
% sigma = std(bfpMat)./size(bfpMat,1);
sigma = cov(bfpMat);

for isubj = 1:5;
    data.subjid = sprintf('F_%d%d%d_%02d',model(1),model(2),model(3),isubj);
    
    theta = mvnrnd(m,sigma);
    %     theta = sigma.*randn(1,size(bfpMat,2))+m;
    
    % generate p_C_hat
    nSamples = 200;
    [~,p_C_hat] = calculate_LL(theta,data,model,[],nSamples);
    
    % generate fake data
    data.resp = rand(length(p_C_hat),1) < p_C_hat;
    
    % plot to make sure it makes sense
    clf
    plot_psychometric_fn(data,6,p_C_hat);
    
    % save
    save(sprintf('data/fitting_data/%s_%s_simple.mat',data.subjid,condition),'theta','p_C_hat','data')
    
    pause;
end

%% check which jobs need to be redone

clear all

condition = 'Ellipse';
subjnumVec = 1:10;
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
    1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
    2 2 1; 2 3 1; ...  % F_O model variants
    2 2 2; 2 3 2];     % F_M model variants

nSubjs = length(subjnumVec);
nModels = size(modelMat,1);

completed = cell(1,nModels);
for itruemodel = 1:nModels;
    truemodel = modelMat(itruemodel,:);
    
    for itestmodel = 1:nModels
        testmodel = modelMat(itestmodel,:);
        
        for isubj = 1:nSubjs
            subjnum = subjnumVec(isubj);
            subjid = sprintf('F_%d%d%d_%02d',truemodel(1),truemodel(2),...
                truemodel(3),subjnum);
            
            load(sprintf('analysis/fits/%s/subj%s_%s_model%d%d%d.mat',condition,...
                subjid,condition,testmodel(1),testmodel(2),testmodel(3)));
            
            completed{itestmodel}(itruemodel,isubj) = length(completedruns);
        end
        
    end
    
end

%% get best fit param
% requires same number of simulated subjects, and all models to be
% simulated and fitted on all simulated subjects

clear all

% condition = 'Line';
% subjnumVec = 1:10;
% modelMat = ...
%     [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
%      1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
%              2 2 1; 2 3 1; ...  % F_O model variants
%              2 2 2; 2 3 2];     % F_M model variants
%
% nSubj = length(subjnumVec);
% nModels = size(modelMat,1);

load('modelfittingsettings.mat')
icond = 1;

condition = conditionVec{icond};


[actualthetaMat, bfpMat, LLMat] = deal(cell(1,nModels)); % organizec by actual model
nParamsVec = nan(1,nModels);
for irealmodel = 1:nModels;
    truemodel = modelMat(irealmodel,:)
    
    bfpMat{irealmodel} = cell(1,nModels);
    LLMat{irealmodel} = nan(nModels,nSubjs);
    for isubj = 1:nSubjs
        subjnum = subjnumVec(isubj)
        subjid = sprintf('F_%d%d%d_%02d',truemodel(1),truemodel(2),...
            truemodel(3),subjnum);
        
        load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition));
        
        if (isubj==1);
            nparams = length(theta);
            nParamsVec(irealmodel) = nparams;
            actualthetaMat{irealmodel} = nan(nSubjs,nparams);
        end
        
        actualthetaMat{irealmodel}(isubj,:) = theta;
        
        for itestmodel = 1:nModels;
            testmodel = modelMat(itestmodel,:);
            
            try
                load(sprintf('analysis/fits/%s/subj%s_%s_model%d%d%d.mat',condition,subjid,condition,testmodel(1),testmodel(2),testmodel(3)))
                
                if (isubj==1);
                    bfpMat{irealmodel}{itestmodel} = nan(nSubjs, size(bfp,2));
                end
                
                idx_posLL = LLVec >0;
                idx_minLL = find(LLVec==min(LLVec(idx_posLL)),1,'first');
                % LLMat{irealmodel}(itestmodel,isubj) = LLVec(idx_minLL);
                LLMat{irealmodel}(itestmodel,isubj) = calculate_LL(bfp(idx_minLL,:),data,testmodel,[],1e3);
                bfpMat{irealmodel}{itestmodel}(isubj,:) = bfp(idx_minLL,:);
                
            catch
                fprintf('test model %d%d%d does not exist for subject %s \n',...
                    testmodel(1),testmodel(2),testmodel(3),subjid)
            end
        end
        
    end
    
end

save(sprintf('analysis/fits/%s/modelrecov_%s.mat',condition,condition),'actualthetaMat','LLMat','bfpMat','nParamsVec','condition','modelMat','subjnumVec')

%% model recovery

clear all

condition = 'Line';
load(sprintf('analysis/fits/%s/modelrecov_%s.mat',condition,condition))
modelnamesVec = {'VVO','VFO','VSO','VVM','VFM','VSM','FFO','FSO','FFM','FSM'};
imodelVec = 1:10;%[1 3];

nSubjs = length(subjnumVec);
nModels = length(imodelVec);
nTrials = 2000;

modelnamesVec = modelnamesVec(imodelVec);
nParamsVec = nParamsVec(imodelVec);
LLMat = LLMat(imodelVec);
LLMat = cellfun(@(x) x(imodelVec,:),LLMat,'UniformOutput',false);

AICMat = cellfun(@(x) 2*bsxfun(@plus,x,nParamsVec'),LLMat,'UniformOutput',false);
AICcMat = cellfun(@(x) bsxfun(@plus,x,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))'),AICMat,'UniformOutput',false);
BICMat = cellfun(@(x) 2*bsxfun(@plus,x,nParamsVec' + log(nTrials)),LLMat,'UniformOutput',false);

confusionMat = nan(nModels);
for irealmodel = 1:nModels;
    
    [m,i]= max(BICMat{irealmodel});
    for iestmodel = 1:nModels;
        confusionMat(irealmodel,iestmodel) = sum(i==iestmodel);
    end
end

figure;
imagesc(confusionMat)
colormap('bone')
ylabel('real model')
xlabel('estimated model')
set(gca,'XTick',1:10,'XTickLabel',modelnamesVec,'YTick',1:10,'YTickLabel',modelnamesVec);

%% plot parameter recovery

clear all

condition = 'Ellipse';
load(sprintf('analysis/fits/%s/modelrecov_%s.mat',condition,condition))

imodel = 10;
model = modelMat(imodel,:);
logflag = getFittingSettings(model,condition);

% get parameter names
counter = 3;
paramnames{1} = 'Jbar high';
paramnames{2} = 'Jbar low';
if strcmp(condition,'Line');
    paramnames{counter} = 'Jbar line';
    counter = counter+1;
end
if (model(1) == 1); % if VP
    paramnames{counter} = 'tau';
    counter = counter+1;
end
if (model(3) == 1) % if optimal decision rule
    if (model(2) == 3) % if assumed same precision
        paramnames{counter} = 'Jbar ellipse, assumed';
        counter = counter+1;
        
        if strcmp(condition,'Line')
            paramnames{counter} = 'Jbar line, assumed';
        end
    end
    
    paramnames{length(paramnames)+1} = 'p change';
else % max rule
    paramnames{length(paramnames)+1} = 'criterion';
end

trueparams = actualthetaMat{imodel};
estparams = bfpMat{imodel}{imodel};

% logging relevant values
trueparams(:,logflag) = log(trueparams(:,logflag));
estparams(:,logflag) = log(estparams(:,logflag));

nparams = size(trueparams,2);

ysubplotsize = floor(sqrt(nparams));
xsubplotsize = ceil(nparams/ysubplotsize);
figure
for iparam = 1:nparams
    subplot(ysubplotsize,xsubplotsize,iparam)
    
    minn = min([trueparams(:,iparam); estparams(:,iparam)]);
    maxx = max([trueparams(:,iparam); estparams(:,iparam)]);
    
    hold on
    plot(trueparams(:,iparam),estparams(:,iparam),'o');
    plot([minn maxx],[minn maxx],'Color',0.7*ones(1,3))
    if logflag(iparam) == 1
        title(sprintf('log %s',paramnames{iparam}))
    else
        title(paramnames{iparam})
    end
    if mod(iparam,xsubplotsize) == 1
        ylabel('estimated value')
    end
    if ((xsubplotsize*ysubplotsize)-iparam < xsubplotsize); xlabel('true value'),end
    
    %     axis equal
    defaultplot
    
end

%% look at LLs of real and estimated parameters

imodel = 1;
model = modelMat(imodel,:);
estLL = LLMat{imodel}(imodel,:);
nSamples = 1000;

[realLL,estLL] = deal(nan(size(estLL)));
for isubj = 1:nSubjs
    
    % load data
    load(sprintf('data/fitting_data/F_%d%d%d_%02d_%s_simple.mat',...
        model(1),model(2),model(3),isubj,condition));
    
    
    % calculate LL
    x = trueparams(isubj,:);
    %     logflag = getFittingSettings(model, condition);
    realLL(isubj) = calculate_LL(x,data,model,[],nSamples);
    estLL(isubj) = calculate_LL(estparams(isubj,:),data,model,[],nSamples);
    
end

figure;
bar(realLL-estLL)
ylabel('positive means actual is better LL')
defaultplot

%% calculate KL divergence
% (run cell above before this one)

% get predictions of best fit param
[~,q_C_hat] = calculate_LL(BFP,data,estmodel);

p_logp = sum(data.resp.*p_C_hat.*log(p_C_hat)) + sum((1-data.resp).*(1-p_C_hat).*log(1-p_C_hat))
p_logq = sum(data.resp.*p_C_hat.*log(q_C_hat)) + sum((1-data.resp).*(1-p_C_hat).*log(1-q_C_hat))
- p_logp - p_logq

