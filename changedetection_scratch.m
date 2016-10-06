%% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% FIGURES
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 



%% didactic figures
clear all

kappa = 15;
xx = linspace(-pi,pi,99);

pchange = circ_vmpdf(linspace(-pi,pi,99),0,2.7);
pchange = pchange./sum(pchange);
% pchange = pchange./sum(pchange);
pnochange = [zeros(1,49) 1 zeros(1,49)];

% p(s_2|s_1)
figure;
plot(xx,pchange,'k'); hold on
plot(xx,pnochange,'Color',0.7*ones(1,3));
xlim([-pi pi])
ylim([0 0.2])
defaultplot
set(gca,'YTick',[0 0.2],'XTick',[-pi 0 pi])

memdist = circ_vmpdf(xx,0,kappa);
memdist = memdist./sum(memdist);

% noisy memory
figure;
plot(xx,memdist);
xlim([-pi pi])
ylim([0 0.2])
defaultplot
set(gca,'YTick',[0 0.2],'XTick',[-pi 0 pi])

post_change = conv(pchange,memdist,'same');
post_nochange = conv(pnochange,memdist,'same');

post_change = post_change./sum(post_change);
post_nochange = post_nochange./sum(post_nochange);

% posterior
figure;
plot(xx,post_change,'k'); hold on
plot(xx,post_nochange,'Color',0.7*ones(1,3));
xlim([-pi pi])
ylim([0 0.2])
defaultplot;
set(gca,'YTick',[0 0.2],'XTick',[-pi 0 pi])

%% decision variable plot (as a function of kappa instead of uncertainty)

kappa_change = 2.6982;

kappabar = @(x,kappa) sqrt(kappa^2 + kappa_change^2 + 2*kappa*kappa_change*cos(x));
d = @(x,kappa) abs(kappa.*cos(x) ...
       +(kappa_change-kappabar(x,kappa)).*log(besseli(0,kappa_change,1)./besseli(0,kappabar(x,kappa),1)));

xx = linspace(0,20,100);
for i = 1:100;
    [decvar(i),fval(i)] = fminsearch(@(x) d(x,xx(i)),rand*3);
end

figure;
plot(xx,abs(decvar))

%% plot decision boundaries for each of the participants for a given model

clear all
modelVec = [1 5 7];
modelStr = {'opt','fix','heurs'};
nModels = length(modelVec);
nSubj = 5;
nCond = 5;


kappa_change = 2.6982;
kappabar = @(x,kappa) sqrt(kappa^2 + kappa_change^2 + 2*kappa*kappa_change*cos(x));
d = @(x,kappa) abs(kappa.*cos(x) ...
       +(kappa_change-kappabar(x,kappa)).*log(besseli(0,kappa_change,1)./besseli(0,kappabar(x,kappa),1)));

decbound_opt = nan(nSubj,nCond); decbound_heurs = decbound_opt;
decbound_fix = nan(1,nSubj);

for imodel = 1:nModels;
    model = modelVec(imodel);
    load(['paramfit_model' num2str(model) '_04292016.mat'])
    bfpp.(modelStr{imodel}) = bestFitParam;
    
    for isubj = 1:nSubj;
        bfp = bestFitParam(isubj,:);
        
        switch model
            case 1
                for icond = 1:nCond;
                    decbound_opt(isubj,icond) = fminsearch(@(x) d(x,bfp(icond)),rand*3);
                end
            case 5
                decbound_fix(isubj) = fminsearch(@(x) d(x,bfp(icond+1)),rand*3);
            case 7
                m = (bfp(7)-bfp(6))./(bfp(5)-bfp(1));
                b = bfp(6)-m.*bfp(1);
                
                for icond = 1:nCond;
                    decbound_heurs(isubj,icond) = m*bfp(icond)+b;
                end
        end
    end
end

linetypeVec = {'-',':','--'};
colorMat = aspencolors(5,'parula');
for isubj = 1:nSubj;
    subplot(2,3,isubj); hold on
    
    for imodel = 1:nModels;
        model = modelVec(imodel);
        
        switch model
            case 1
                decbounds = decbound_opt;
            case 5
                decbounds = repmat(decbound_fix',1,nCond);
            case 7
                decbounds = decbound_heurs;
        end
        
        plot(bfpp.(modelStr{imodel})(isubj,1:nCond),decbounds(isubj,:),['k' linetypeVec{imodel}])
        for icond = 1:nCond;
            plot(bfpp.(modelStr{imodel})(isubj,icond),decbounds(isubj,icond),'.','MarkerSize',24,'Color',colorMat(icond,:));
        end
                
    end

end

%% plot indvl subject fits

model = 17;
aveplot = 0;
subjids = {'1','3','4','EL','ND'};
nSubj = length(subjids);

if model < 10;
    load(['paramfit_model' num2str(model) '_04292016.mat'],'bestFitParam','nLL_est')
else
load(['paramfit_model' num2str(model) '_09282016.mat'],'bestFitParam','nLL_est')
end

figure; 
if ~(aveplot)
    for isubj = 1:nSubj;
        subplot(2,3,isubj)
        plot_datafit(subjids{isubj}, model, bestFitParam(isubj,:))
    end
else
    plot_datafit(subjids,model,bestFitParam)
end

%% plot of unparameterized and powerlaw kappas 
clear all
figure

model = 5;

% nonparemterized one
load(['paramfit_model' num2str(model) '_04292016.mat'],'bestFitParam')
kappaVec1 = bestFitParam(:,1:5);

% pwer law one
model = str2double(['1' num2str(model)]);
load(['paramfit_model' num2str(model) '_09282016.mat'],'bestFitParam')

eccentricities = linspace(0.15,.999,50);%[0.15 0.3 0.5 0.8 0.999];
ecc_low = min(eccentricities);
ecc_hi = max(eccentricities);

nSubj = 5;
kappaVec2 = nan(nSubj,50);
for isubj = 1:nSubj;
    theta = bestFitParam(isubj,:);
    kappa_c_low = exp(theta(1));
    kappa_c_high = exp(theta(2));
    beta = theta(3);
    
    alpha = ((1/kappa_c_low)-(1/kappa_c_high))/(ecc_low^-beta - ecc_hi^-beta);
    kappaVec2(isubj,:) = 1./((1/kappa_c_low)+ alpha*(eccentricities.^-beta - ecc_low^-beta));
    
    subplot(2,3,isubj);
    plot([0.15 0.3 0.5 .8 .999],kappaVec1(isubj,:),'o'); hold on;
    plot(eccentricities,kappaVec2(isubj,:),'-');
    defaultplot
end



%% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
%               ANALYSES
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 





%% AIC and BIC of human subjects

clear all
modelVec = [1 5 7];
nModels = length(modelVec);
nSubj = 5;

nLLMat = nan(nModels,nSubj);
for imodel = 1:nModels;
    model = modelVec(imodel);
    
    if model < 10;
        load(['paramfit_model' num2str(model) '_04292016.mat'])
    else
        load(['paramfit_model' num2str(model) '_09282016.mat'])
    end
    
    nLLMat(imodel,:) = nLL_est;
    AICMat(imodel,:) = 2*(nLL_est + size(bestFitParam,2));
    BICMat(imodel,:) = 2*nLL_est + size(bestFitParam,2)*log(3010);
end

figure;
bar(bsxfun(@minus,AICMat(3,:),AICMat)')
figure;
bar(bsxfun(@minus,BICMat(3,:),BICMat)')
%%
clear all

xx = linspace(0,20,100);
alpha = 0.15;
yy = exp(xx.*alpha);

plot(xx,yy);
pause
plot(xx,1./(yy.^2))
%% percent correct for each reliability (across subjects)

clear all
colorMat = aspencolors(5,'parula');

subjids = {'1','3','4','EL','ND'};
nSubj = length(subjids);
nCond = 5;

PC = nan(nSubj,nCond);
for isubj = 1:nSubj;
    subjid = subjids{isubj};
    
    [data] = concatdata(subjid,'Detection');
    Xdet = conditionSeparator(data,0);
    PC(isubj,:) = cell2mat(cellfun(@(x) mean(x(:,2)),Xdet,'UniformOutput',false));
end

mean_PC = mean(PC);
sem_PC = std(PC)/sqrt(nSubj);

figure;hold on
for icond = 1:nCond;
    errorbar(icond,mean_PC(icond),sem_PC(icond),'Color',colorMat(icond,:));
end
defaultplot;
ylim([0 1])
set(gca,'XTick',1:5,'XTickLabel',1:5,...
    'YTick',[0 0.5 1])
xlabel('reliability')
ylabel('proportion correct')

%% full psychometric function (across subjects)
clear all

subjids = {'1','3','4','EL','ND'};
nSubj = length(subjids);

colorMat = aspencolors(5,'parula');

stimLevels = cell(1,5); trialNums = cell(1,5); nResps = cell(1,5);
for isubj = 1:nSubj;
    subjid = subjids{isubj};
    
    [data] = concatdata(subjid,'Detection');
    [stimlevels, trialnums, nresps] = conditionSeparator(data,1);
    nLevels = length(stimlevels{1});
    
    stimLevels = cellfun(@(x,y) [x;y],stimLevels,stimlevels,'UniformOutput',false);
    trialNums = cellfun(@(x,y) [x;y],trialNums,trialnums,'UniformOutput',false);
    nResps = cellfun(@(x,y) [x;y],nResps,nresps,'UniformOutput',false);
    
end

mean_stimLevels = cellfun(@(x) mean(x),stimLevels,'UniformOutput',false);
mean_pResps = cellfun(@(x,y) mean(x./y),nResps,trialNums,'UniformOutput',false);
sem_pResps = cellfun(@(x,y) std(x./y)/sqrt(nSubj),nResps,trialNums,'UniformOutput',false);

figure; hold on;
for icond=1:5
    errorbar(mean_stimLevels{icond},mean_pResps{icond},sem_pResps{icond},'Color',colorMat(icond,:));
end
% currplot = plot(stimlevels, nresps./trialnums,'.-');
% set(currplot,'Color',       colorMat(cond,:),...
%     'MarkerSize',24);
defaultplot;
axis([-45 45 0 1])
set(gca,'Ytick',[0 0.5 1],'Xtick',[-45 0 45])
ylabel('p(respond "no change")');
xlabel('orientation change (deg)');

%% marginal psychometric function (across all subjects)
nLevels = length(mean_stimLevels{1});
nCond = 5;
mean_stimlevels = mean(reshape(cell2mat(mean_stimLevels),nLevels,nCond)');
mean_presps = mean(reshape(cell2mat(mean_pResps),nLevels,nCond)');
sem_presps = std(reshape(cell2mat(mean_pResps),nLevels,nCond)')./sqrt(nCond);

figure; 
errorbar(mean_stimlevels,mean_presps,sem_presps,'k');
defaultplot
axis([-45 45 0 1])
set(gca,'Ytick',[0 0.5 1],'Xtick',[-45 0 45])
ylabel('p(respond "no change")');
xlabel('orientation change (deg)');
%% full psychometric function (indvl subject)

subjid = '4';

[data] = concatdata(subjid,'Detection');
[stimlevels, trialnums, nresps] = conditionSeparator(data,1);


colorMat = aspencolors(5,'parula');

conditions = 1:5;
nCond = 5;


hold on
for icond = 1:nCond;
    cond = conditions(icond);
    
    currplot = plot(stimlevels{cond}, nresps{cond}./trialnums{cond},'.-');
    set(currplot,'Color',       colorMat(cond,:),...
        'MarkerSize',24);
    defaultplot;
    axis([-45 45 0 1])
    set(gca,'Ytick',[0 0.5 1],'Xtick',[-45 0 45])
    ylabel('p(respond same)');
    xlabel('orientation change (deg)');
    
end
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
%
% nSubj = 10;
% subjids = num2cell(1:nSubj);
% subjids = cellfun(@(x) [num2str(x)],subjids,'UniformOutput',false);
%
% simulate_realisticdata(model,subjids)

%% remove text spaces for each subject number and get best fit parameters and save

subjids = {'1','2','3','4','5','6','7','8','9','10'};
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
        titleVec = {'\kappa 1','\kappa 2','\kappa 3','\kappa 4','\kappa 5','\lambda'};
        subplotsize = 3;
    case 5
        titleVec = {'\kappa 1','\kappa 2','\kappa 3','\kappa 4','\kappa 5','\kappa_{assumed}','\lambda'};
        subplotsize = 4;
    case 7
        titleVec = {'\kappa 1','\kappa 2','\kappa 3','\kappa 4','\kappa 5','slope','intercept','lambda'};
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

modelVec = [ 1 5 7];
nModels = length(modelVec);

%% get matrix of nLLs
filepath = 'analysis/3_fakedata/fakedata/fits/';

AICconfusionmatrix = zeros(nModels,nModels);
BICconfusionmatrix = AICconfusionmatrix;
nLLMat = nan(10,nModels,nModels);
for itruemodel = 1:nModels;
    truemodel = modelVec(itruemodel);
    
    subjids = {'1','2','3','4','5','6','7','8','9','10'};
    subjids = cellfun(@(x) ['F' num2str(truemodel) x],subjids,'UniformOutput',false);
    nSubj = length(subjids);
    
    AICMat = nan(nSubj,nModels);
    BICMat = AICMat;
    for itestmodel = 1:nModels;
        testmodel = modelVec(itestmodel);
        
        [bestFitParam, nLL_est] = combineparamfit(testmodel,subjids,filepath);
        nLLMat(:,itestmodel,itruemodel) = nLL_est;
        AICMat(:,itestmodel) = 2*nLL_est(:)+2*size(bestFitParam,2);
        BICMat(:,itestmodel) = 2*nLL_est(:)+size(bestFitParam,2)*log(3010);
    end
    
    for isubj = 1:nSubj;
        BICconfusionmatrix(itruemodel,:) = BICconfusionmatrix(itruemodel,:) +...
            (BICMat(isubj,:) == min(BICMat(isubj,:)));
        AICconfusionmatrix(itruemodel,:) = AICconfusionmatrix(itruemodel,:) +...
            (AICMat(isubj,:) == min(AICMat(isubj,:)));
    end
end

AICconfusionmatrix
BICconfusionmatrix


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

%% POWER LAW MAPPING (eccentricity --> sigma)

% figure

log_sigma_c_low = 2; % log sigma for lowest contrast
log_sigma_c_hi = -2; % log sigma for highest contrast. don’t let this go below -4, it may cause you numerical problems

beta = 10; % curvature
sigma_c_low = exp(log_sigma_c_low);
sigma_c_hi = exp(log_sigma_c_hi);
kappa_c_low = 1./(sigma_c_low^2)
kappa_c_hi = 1./(sigma_c_hi^2)

eccentricities = linspace(.15,.999,100);

c_low = min(eccentricities);
c_hi = max(eccentricities);
alpha = ((1/kappa_c_low)-(1/kappa_c_hi))/(c_low^-beta - c_hi^-beta);
sigmas = sqrt((1/kappa_c_low)+ alpha*(eccentricities.^-beta - c_low^-beta));
% kappas = (c_hi^-beta - c_low^-beta)./(((1/kappa_c_low)*(c_hi^-beta - eccentricities.^-beta)) - ((1/kappa_c_hi)*(eccentricities.^-beta - c_low^-beta)));
kappas = (c_hi^-beta - c_low^-beta)./((sigma_c_low^2*(c_hi^-beta - eccentricities.^-beta)) - (sigma_c_hi^2*(eccentricities.^-beta - c_low^-beta)));

plot(eccentricities,1./(sigmas.^2),'k-'); 
% hold on
% plot(eccentricities,kappas,'b:');
%% POWER LAW MAPPING (eccentricity --> kappa)

figure

log_kappa_c_low = -4; % log sigma for lowest contrast
log_kappa_c_hi = 1; % log sigma for highest contrast. don’t let this go below -4, it may cause you numerical problems

beta = 1; % curvature
kappa_c_low = exp(log_kappa_c_low);
kappa_c_hi = exp(log_kappa_c_hi);

eccentricities = linspace(.15,.999,100);

c_low = min(eccentricities);
c_hi = max(eccentricities);
kappas = (c_hi^-beta - c_low^-beta)./ ...
    ((1/kappa_c_low).*(c_hi^-beta - eccentricities.^-beta) - (1/kappa_c_hi).*(eccentricities.^-beta - c_low^-beta));
% alpha = (kappa_c_low^2-kappa_c_hi^2)/(c_low^-beta - c_hi^-beta);
% sigmas = sqrt(kappa_c_low^2 + alpha*(eccentricities.^-beta - c_low^-beta));

plot(eccentricities,kappas,'k-')