
%% load subject data

subjid = {'1'};
[data] = concatdata(subjid,'Detection');
% [Xdet] = conditionSeparator(data);
% nCond = length(Xdet);

mean(data(data(:,1)~=0,1))
std(data(data(:,1)~=0,1))
%% remove text spaces for each subject number;
model = 7;

subjnum = 1;
subjids = {'1','2','3','4','5','6','7','8','9','10'};
subjids = cellfun(@(x) [ num2str(subjnum) '_' x '0000'],subjids,'UniformOutput',false);

for isubj = 1:length(subjids);
    subjid = subjids{isubj};
    removetxtspaces(model,subjid)
end

%% get best fit parameters for all subjects and save
clear all
model = 1;
[bestFitParam, nLL_est, subjids] = combineparamfit(model);
save(['analysis/4_realdata/fits/paramfit_model' num2str(model) '.mat'])

%% plot real data parameter fits
clear;
subjidVec = {'1','3','4','EL','ND'};
model = 7;
indvlplot = 0;

load(['analysis/4_realdata/fits/paramfit_model' num2str(model) '_04292016.mat']);

figure;
if (indvlplot)
    for isubj = 1:length(subjidVec);
        subplot(2,3,isubj)
        plot_datafit(subjidVec{isubj}, model, bestFitParam(isubj,:))
    end
else
    plot_datafit(subjidVec, model, bestFitParam)
end

%% calculate likelihood of real data
subjids = {'1','3','4','EL','ND'};
model = 7;

switch model
    case 1
        logflag = [ones(1,5) 0];
    case 5
        logflag = [ones(1,5) 1 0];
    case 7
        logflag = [ones(1,5) zeros(1,2) 0];
end
% bfp(:,logical(logflag)) = log(bfp(:,logical(logflag)));

bestFitParam(:,logical(logflag)) = log(bestFitParam(:,logical(logflag)));
for isubj = 1:length(subjids);
    subjid = subjids{isubj};
    
    % load data
    [data] = concatdata(subjid,'Detection');
    [Xdet] = conditionSeparator(data);
    
    ll(isubj) = loglike(Xdet, model, bestFitParam(isubj,:));
end

ll

%% checking two concentration parameter values
pop1 =  std(circ_vmrnd(0,2.6982, [10000,1000])*90/pi);
pop2 =  std(circ_vmrnd(0,8.742, [10000,1000])*180/pi);

[count1,centers1] = hist(pop1);
[count2,centers2] = hist(pop1);

figure;
plot([centers1; centers2],[count1; count2])

%% ====================================
%        PARAMETER RECOVERY
%  ====================================
clear all

model = 7;
%% simulate fake data

nSubj = 10;
subjids = num2cell(1:nSubj);
subjids = cellfun(@(x) [num2str(x)],subjids,'UniformOutput',false);

simulate_realisticdata(model,subjids)

%% remove text spaces for each subject number and get best fit parameters and save

subjids = {'1','2','3','4','5'};
subjids = cellfun(@(x) ['F' num2str(model) x],subjids,'UniformOutput',false);
filepath = 'analysis/3_fakedata/fakedata/fits/';
nSubj = length(subjids);

for isubj = 1:nSubj;
    subjid = subjids{isubj};
    removetxtspaces(model,subjid,filepath)
end

[bestFitParam, nLL_est, subjids] = combineparamfit(model,subjids,filepath);
save([filepath 'paramfit_model' num2str(model) '.mat'],'bestFitParam','nLL_est','subjids')

%% load actual theta values

nParams = size(bestFitParam,2);
switch model
    case 1
        logflag = [ones(1,5) 0];
    case 5
        logflag = [ones(1,5) 1 0];
    case 7
        logflag = [ones(1,5) zeros(1,2) 0];
end


thetaMat = nan(nSubj,nParams);
LL_trueish = nan(1,nSubj);
LL_estimate = LL_trueish;
for isubj = 1:nSubj;
    subjid = subjids{isubj};
    load(['fakedata_subj' subjid '.mat'],'theta','Xdet')
    thetaMat(isubj,:) = theta;
    theta(logical(logflag)) = log(theta(logical(logflag)));
    LL_trueish(isubj) = loglike(Xdet, model, theta);
    
    % best fit parameter LL
    theta_est = bestFitParam(isubj,:);
    theta_est(logical(logflag)) = log(theta_est(logical(logflag)));
    LL_estimate(isubj) = loglike(Xdet,model,theta_est);
end

LL_trueish
LL_estimate
nLL_est
%% plot data model predictions for true and fake data

figure;

for isubj = 1:nSubj;
% real parameters
subplot(2,5,isubj)
theta  = thetaMat(isubj,:);
plot_datafit(subjids{isubj}, model, theta)
title('actual \theta')

% fake parameters
subplot(2,5,nSubj+isubj)
theta = bestFitParam(isubj,:);
plot_datafit(subjids{isubj}, model, theta)
title('estimated \theta')
end


%% plot parameter recovery plots
switch model
    case 1
        titleVec = {'1','2','3','4','5','lapse'};
        subplotsize = 3;
    case 5
        titleVec = {'1','2','3','4','5','perceived','lapse'};
        subplotsize = 4;
    case 7
        titleVec = {'1','2','3','4','5','slope','intercept','lapse'};
        subplotsize = 4;
end

figure; hold on
for iparam = 1:nParams;
    subplot(2,subplotsize,iparam);
    plot(thetaMat(:,iparam),bestFitParam(:,iparam),'ko'); hold on
    
    maxplot = max([thetaMat(:,iparam); bestFitParam(:,iparam)]);
    plot([0 maxplot],[0 maxplot],'--','Color',0.7*ones(1,3));
    
    defaultplot
    xlabel('actual')
    ylabel('estimated')
    title(titleVec{iparam})
end

%% plot nLLs
% if this is working, estimated nLLs should be around the diagonal
% bad if the estimated is better than actual (would be lower than diagonal

figure;
plot(-LL_trueish,nLL_est,'ko');hold on;
maxplot = max([-LL_trueish,nLL_est]);
plot([0 maxplot],[0 maxplot],'--','Color',0.7*ones(1,3));
defaultplot
xlabel('actual')
ylabel('estimated')





%% ==============================================
%                MODEL RECOVERY
% ===============================================
clear all; 

modelVec = [5 7 1];
nModels = length(modelVec);

%% get matrix of nLLs
filepath = 'analysis/3_fakedata/fakedata/fits/';

AICconfusionmatrix = zeros(nModels,nModels);
BICconfusionmatrix = AICconfusionmatrix;
for itruemodel = 1:nModels;
    truemodel = modelVec(itruemodel);
    
    subjids = {'1','2','3','4','5'};
    subjids = cellfun(@(x) ['F' num2str(truemodel) x],subjids,'UniformOutput',false);
    nSubj = length(subjids);
    
    AICMat = nan(nSubj,nModels);
    BICMat = AICMat;
    for itestmodel = 1:nModels;
        testmodel = modelVec(itestmodel);
        
        [bestFitParam, nLL_est] = combineparamfit(testmodel,subjids,filepath);
        AICMat(:,itestmodel) = 2*nLL_est(:)+2*size(bestFitParam,1);
        BICMat(:,itestmodel) = 2*nLL_est(:)+size(bestFitParam,1)*log(3010);
    end
    
    for isubj = 1:nSubj; 
        BICconfusionmatrix(itruemodel,:) = BICconfusionmatrix(itruemodel,:) +...
            (BICMat(isubj,:) == min(BICMat(isubj,:)));
        AICconfusionmatrix(itruemodel,:) = AICconfusionmatrix(itruemodel,:) +...
            (AICMat(isubj,:) == min(AICMat(isubj,:)));
    end
end


%% ==============================================
%                   ANALYSES
% ===============================================

%% calculating K-L divergence between model and human data
clear all

modelVec = [1 5 7];
subjVec = {'1','3','4','EL','ND'};

nModels = length(modelVec);
nSubj = length(subjVec);

gamma = 0.577215; % euler's constant
idx = repmat(1:155,2,1);
A = 2./(1:2:309);
G = -gamma - log(2) + cumsum(A);
G = [0 -gamma-log(2) G(idx(:))];

KLMat = nan(nModels,nSubj);
for isubj = 1:nSubj;
    subjid = subjVec{isubj};
    
    % load data
    [data] = concatdata(subjid,'Detection');
    [Xdet] = conditionSeparator(data);
    nCond = length(Xdet);
    
    nBins = 10;
    entr = nan(1,nCond);
    % estimating entropy p*log(p)
    for icond = 1:nCond;
        blah = Xdet{icond};
        
        % grassberger unbiased approximation
        nresp1 = sum(blah(blah(:,1)==0,2));
        ntrials = size(blah(blah(:,1)==0,:),1);
        entr(icond) = ntrials*(G(ntrials) - 1/ntrials*(nresp1*G(nresp1+1) + (ntrials-nresp1)*G(ntrials-nresp1+1)));
        
        %         % biased calculation
        %         presp_nochange = mean(blah(blah(:,1)==0,2));
        %         entr(icond) = presp_nochange*log(presp_nochange) + (1-presp_nochange)*log(1-presp_nochange);
        %         entr(icond) = entr(icond).*size(blah(blah(:,1) == 0,1),1);
        
        changetrials = sortrows(abs(blah(blah(:,1)~=0,:)));
        nchangeTrials = size(changetrials,1);
        nTrialsperBin = floor(nchangeTrials/nBins);
        leftover = nchangeTrials - nTrialsperBin*nBins;
        
        %         % biased
        %         presp_quantile = [mean(changetrials(1:nTrialsperBin+leftover,2)) nan(1,nBins-1)];
        %         for ibin = 2:nBins;
        %             presp_quantile(ibin) = mean(changetrials((ibin-1)*nTrialsperBin+leftover+1:ibin*nTrialsperBin+leftover,2));
        %         end
        %         nTrialsbinVec = [nTrialsperBin+leftover nTrialsperBin*ones(1,nBins-1)];
        %         nTrialsbinVec((presp_quantile == 0) | (presp_quantile == 1)) = [];
        %         presp_quantile((presp_quantile == 0) | (presp_quantile == 1)) = [];
        %
        %         entr(icond) = entr(icond) + nTrialsbinVec*(presp_quantile.*log(presp_quantile) + (1-presp_quantile).*log(1-presp_quantile))';
        
        % using grassberger estimator
        nTrialsbinVec = [nTrialsperBin+leftover nTrialsperBin*ones(1,nBins-1)];
        trialscumsum = [0 cumsum(nTrialsbinVec)];
        for ibin = 1:nBins;
            ntrials = nTrialsbinVec(ibin);
            nresp1 = sum(changetrials((trialscumsum(ibin)+1):trialscumsum(ibin+1),2));
            entrr(ibin) = ntrials*(G(ntrials) - 1/ntrials*(nresp1*G(nresp1+1) + (ntrials-nresp1)*G(ntrials-nresp1+1)));
        end
        entr(icond) = - entr(icond) - sum(entrr);
        
        %     plot(presp_quantile,'ko'); defaultplot; pause
    end
    entr = sum(entr) % mean entropy over conditions
    
    for imodel = 1:nModels;
        model = modelVec(imodel);
        
        % load model parameters
        load(['paramfit_model' num2str(model) '.mat']) % load model
        bleh = cellfun(@(x) strcmp(x,subjid),subjids, 'UniformOutput',false);
        subjidx = find(cell2mat(bleh) == 1);
        theta = bestFitParam(subjidx,:);
        bias = 0;
        
        % calculating log likelihood q*log(p)
        theta(1:nCond) = log(theta(1:nCond));
        if model == 5; theta(nCond+1) = log(theta(nCond+1)); end
        [ll] = loglike(Xdet, model, theta);
        
        KLMat(imodel,isubj) = entr-ll;
    end
    
end
%% seeing which model wins as a function of nTrials (NN data)

modelVec = [1 5 7];
subjVec = 1:5;

nModels = length(modelVec);
nSubjs = length(subjVec);

AICMat = nan(nModels,nSubjs,10);
nParamsVec = nan(1,nModels);
count = zeros(10,3);
for isubj = 1:nSubjs;
    subjnum = subjVec(isubj);
    
    subjids = {'1','2','3','4','5','6','7','8','9','10'};
    subjids = cellfun(@(x) [ num2str(subjnum) '_' x '0000'],subjids,'UniformOutput',false);
    
    for imodel = 1:nModels;
        model = modelVec(imodel);
        [bestFitParam, nLL_est] = combineparamfit(model,subjids);
        AICMat(imodel,isubj,:) = nLL_est + 2*size(bestFitParam,2);
    end
    
    for i = 1:10;
        count(i,:) = count(i,:) + (AICMat(:,isubj,i) == min(AICMat(:,isubj,i)))';
    end
    
    nParamsVec(imodel) = size(bestFitParam,2);
end

% making plot for which model wins
figure; hold on
fill([1:10 10:-1:1],[5*ones(1,10) zeros(1,10) ],aspencolors('green')); % optimal
fill([1:10 10:-1:1],[sum(count(:,2:3)') zeros(1,10)],aspencolors('salmon')); % heuristic
fill([1:10 10:-1:1],[count(:,2)' zeros(1,10)],aspencolors('greyblue')); % fixed


% AICMat = 2*nParamsVec + nLLMat;