
%%
clear all

subjidVec = {'S02','S03','S06','S08','S10','S11','S14',...
    'S15','S16','S17','S19','S20','S23'};
nSubj = length(subjidVec);
nBins = 6;
quantilebinning=1;

h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

figure(1);
clf

% ============= ELLIPSE ===========
condition = 'Ellipse';
[x_mean, pc_data] = deal(nan(5,nBins,nSubj));
[HRallVec,HRlowVec,HRhighVec,FARVec] = deal(nan(nSubj,5));
for isubj = 1:nSubj;
    subjid = subjidVec{isubj};
    
    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')
    
    % get psychometric function binned data
    [x_mean(:,:,isubj), pc_data(:,:,isubj)] = plot_psychometric_fn(data,nBins,[],quantilebinning);
    
    % get hits/false alarms binned data
        [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:)] = ...
            plot_HR_FAR(data,[],0);
end

% get participant and model means: psychometric fn
xrange = nanmean(x_mean,3);
partM = nanmean(pc_data,3);
partSEM = nanstd(pc_data,[],3)./sqrt(nSubj-1);
save('subjpsychometricfn.mat','xrange','partM','partSEM')

% get participant and model means: hits/false alarms
m_HRall = mean(HRallVec);
m_HRlow = mean(HRlowVec);
m_HRhigh = mean(HRhighVec);
m_FAR = mean(FARVec);
sem_HRall = std(HRallVec)./sqrt(nSubj);
sem_HRlow = std(HRlowVec)./sqrt(nSubj);
sem_HRhigh = std(HRhighVec)./sqrt(nSubj);
sem_FAR = std(FARVec)./sqrt(nSubj);

% PLOT MODEL FITS
subplot(1,4,1), hold on;
xlim([-0.2 pi/2+0.2])
ylim([0 1])
for ii = 1:5;
    plot(xrange(ii,:),partM(ii,:),'Color',colorMat(ii,:))
    errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat(ii,:))
end
set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
defaultplot
    xlabel('magnitude change')
%     ylabel('proportion respond change')

% PLOT HITS FALSE ALARMS
subplot(1,4,3), hold on;
xlim([-0.5 4.5])
ylim([0 1])
plot(0:4,m_HRall,'Color',colorMat(1,:))
plot(0:4,m_HRlow,'Color',colorMat(2,:))
plot(0:4,m_HRhigh,'Color',colorMat(3,:))
plot(0:4,m_FAR,'Color',colorMat(4,:))
errorb(0:4,m_HRall,sem_HRall,'color',colorMat(1,:))
errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat(2,:))
errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat(3,:))
errorb(0:4,m_FAR,sem_FAR,'color',colorMat(4,:))
set(gca,'XTick',0:4,'XTickLabel',0:4)
defaultplot
xlabel('number of high reliability ellipses')
%     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')

figure(2);
% ============= LINE ===========
condition = 'Line';
[x_mean, pc_data] = deal(nan(5,nBins,nSubj));
[HRallVec,HRlowVec,HRhighVec,FARVec] = deal(nan(nSubj,5));
for isubj = 1:nSubj;
    subjid = subjidVec{isubj};
    
    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')
    
    % get psychometric function binned data
    [x_mean(:,:,isubj), pc_data(:,:,isubj)] = plot_psychometric_fn(data,nBins,[],quantilebinning);
    
    % get hits/false alarms binned data
        [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:)] = ...
            plot_HR_FAR(data,[],0);
end

% get participant and model means: psychometric fn
xrange = nanmean(x_mean,3);
partM = nanmean(pc_data,3);
partSEM = nanstd(pc_data,[],3)./sqrt(nSubj-1);
save('subjpsychometricfn.mat','xrange','partM','partSEM')

% get participant and model means: hits/false alarms
m_HRall = mean(HRallVec);
m_HRlow = mean(HRlowVec);
m_HRhigh = mean(HRhighVec);
m_FAR = mean(FARVec);
sem_HRall = std(HRallVec)./sqrt(nSubj);
sem_HRlow = std(HRlowVec)./sqrt(nSubj);
sem_HRhigh = std(HRhighVec)./sqrt(nSubj);
sem_FAR = std(FARVec)./sqrt(nSubj);

figure(1);
% PLOT MODEL FITS
subplot(1,4,2), hold on;
xlim([-0.2 pi/2+0.2])
ylim([0 1])
for ii = 1:5;
    plot(xrange(ii,:),partM(ii,:),'Color',colorMat(ii,:))
    errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat(ii,:))
end
set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
defaultplot
    xlabel('magnitude change')
%     ylabel('proportion respond change')

% PLOT HITS FALSE ALARMS
subplot(1,4,4), hold on;
xlim([-0.5 4.5])
ylim([0 1])
plot(0:4,m_HRall,'Color',colorMat(1,:))
plot(0:4,m_HRlow,'Color',colorMat(2,:))
plot(0:4,m_HRhigh,'Color',colorMat(3,:))
plot(0:4,m_FAR,'Color',colorMat(4,:))
errorb(0:4,m_HRall,sem_HRall,'color',colorMat(1,:))
errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat(2,:))
errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat(3,:))
errorb(0:4,m_FAR,sem_FAR,'color',colorMat(4,:))
set(gca,'XTick',0:4,'XTickLabel',0:4)
defaultplot
xlabel('number of high reliability ellipses')
%     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')


%% DATA alone plot

clear all

load('subjpsychometricfn.mat')

h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

% PLOT MODEL FITS
subplot(1,2,1), hold on;
xlim([-0.2 pi/2+0.2])
ylim([0 1])
for ii = 1:5;
    errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat(ii,:))
end
set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
defaultplot
    xlabel('magnitude change')
%     ylabel('proportion respond change')

% PLOT HITS FALSE ALARMS
subplot(1,2,2), hold on;
xlim([-0.5 4.5])
ylim([0 1])
errorb(0:4,m_HRall,sem_HRall,'color',colorMat(1,:))
errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat(2,:))
errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat(3,:))
errorb(0:4,m_FAR,sem_FAR,'color',colorMat(4,:))
set(gca,'XTick',0:4,'XTickLabel',0:4)
defaultplot
xlabel('number of high reliability ellipses')
%     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')



%% ALL SUBJ MODEL FITS: psychometric function and hits/false alarms

clear all
condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari/';
additionalpaths2 = '';%'_keshvari';
% condition = 'combined';
% additionalpaths = '';%'combined_diffdisp/';%'ellipse_keshvari/';

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
    1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
    2 2 1; 2 3 1; ...  % F_O model variants
    2 2 2; 2 3 2];     % F_M model variants
% imodelVec = [1 2 3 7 8];
imodelVec = [4 5 6 9 10];

load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition));
modelnames = {  'VVO', 'VFO', 'VSO',...
    'VVM', 'VFM', 'VSM',...
    'FFO', 'FSO',...
    'FFM', 'FSM'};
nSubj = size(LLMat,2);
nTrials = 2000;

% calculated AIC, AICc, and BIC
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials)); % BIC
BICMat = bsxfun(@minus,BICMat,BICMat(1,:)); % BIC relative to VVO

% mean sem
m = nanmean(BICMat,2);
sem = nanstd(BICMat,[],2)/sqrt(nSubj);

nModels = length(imodelVec);
for imodel = 1;%:nModels;%imodelVec%1:2;%nModels
    modelnum = imodelVec(imodel);
    model = modelMat(modelnum,:);
    
    % load bfp fits
%     load(sprintf('analysis/fits/%sbfp_%s%s.mat',additionalpaths,condition,additionalpaths2))
    bfpmat = bfpMat{modelnum};
    nSubj = length(subjidVec);
    
    % prediction stuff
    nSamples = 50;
    nBins = 6;
    quantilebinning=1;
    
    figure(1);
    [x_mean, pc_data, pc_pred] = deal(nan(5,nBins,nSubj));
    [HRallVec,HRlowVec,HRhighVec,FARVec,mod_HRallVec,mod_HRlowVec,mod_HRhighVec,mod_FARVec] = deal(nan(nSubj,5));
    for isubj = 1:nSubj
        subjid = subjidVec{isubj};
        bfp = bfpmat(isubj,:);
        
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
    
    save('subjpsychometricfn.mat','xrange','partM','partSEM',...
        'm_HRall','m_HRlow','m_HRhigh','m_FAR','sem_HRall','sem_HRlow',...
        'sem_HRhigh','sem_FAR')
    
    % get colormap info
%     colorMat = ...
%         [223 171 110;...
%          212  95 100;...
%          169 109 223;...
%          126 159 235;...
%          117 214 180]./255;
    h = figure(99);
    cmap = colormap('parula'); % get a rough colormap
    close(h)
    idxs = round(linspace(1,size(cmap,1),5));
    colorMat = cmap(idxs,:);
    
    % plot
    figure(2);
    
    % PLOT MODEL FITS
    subplot(nModels,3,3*imodel-2); hold on
    xlim([-0.2 pi/2+0.2])
    for ii = 1:5;
        plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
            modelSEM(ii,:),colorMat(ii,:),colorMat(ii,:))
    end
    set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
    %     xlabel('magnitude change')
    %     ylabel('proportion respond change')
    
    % PLOT HITS FALSE ALARMS
    subplot(nModels,3,3*imodel-1); hold on;
    xlim([-0.5 4.5])
    ylim([0 1])
    plot_summaryfit(0:4,m_HRall,sem_HRall,m_mod_HRall,sem_mod_HRall,colorMat(1,:),colorMat(1,:));
    plot_summaryfit(0:4,m_HRlow,sem_HRlow,m_mod_HRlow,sem_mod_HRlow,colorMat(2,:),colorMat(2,:));
    plot_summaryfit(0:4,m_HRhigh,sem_HRhigh,m_mod_HRhigh,sem_mod_HRhigh,colorMat(3,:),colorMat(3,:));
    plot_summaryfit(0:4,m_FAR,sem_FAR,m_mod_FAR,sem_mod_FAR,colorMat(4,:),colorMat(4,:));
    set(gca,'XTick',0:4,'XTickLabel',0:4)
    %     xlabel('number of high reliability ellipses')
    %     ylabel('proportion respond change')
    %     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')
    
    subplot(nModels,3,3*imodel); hold on
    fill([m(modelnum)-sem(modelnum) m(modelnum)+sem(modelnum) m(modelnum)+sem(modelnum)...
        m(modelnum)-sem(modelnum)],[0 0 nSubj+1 nSubj+1],0.7*ones(1,3));
    barh(BICMat(modelnum,:),'k')
    xlim([-200 2000])
    % xlim([0.5 nModels+0.5])
    set(gca,'YTick',[],'YTickLabel',[]);
    defaultplot
    
end

%% model comparison (AICc and BIC)

clear all
close all

condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari/';%'';%'combined_diffdisp/';%
load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition));
% modelnames = {'VVO','VVM','VSO','VSM'};
modelnames = {  'VVO', 'VFO', 'VSO',...
                'VVM', 'VFM', 'VSM',...
                       'FFO', 'FSO',...
                       'FFM', 'FSM'};
nModels = length(modelnames);
nTrials = 2000;

% calculated AIC, AICc, and BIC
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));
nSubj = size(BICMat,2);

% figure;
% bar(BICMat)
% title('BIC')
% xlim([0.5 nModels+0.5])
% set(gca,'XTick',1:10,'XTickLabel',modelnames);
% defaultplot

% relatve to VVO

% AICcMat = bsxfun(@minus,AICcMat,AICcMat(1,:));
BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

% figure;
% bar(BICMat)
% title('BIC')
% xlim([0.5 nModels+0.5])
% set(gca,'XTick',1:10,'XTickLabel',modelnames);
% defaultplot

% mean sem of same thing
M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

figure; hold on
for imodel = 1:9
    fill([-0.5 0.5 0.5 -0.5]+imodel,[-SEM_BIC(imodel+1) -SEM_BIC(imodel+1) ...
        SEM_BIC(imodel+1) SEM_BIC(imodel+1)]+M_BIC(imodel+1),0.7*ones(1,3),'LineStyle','none')
end
bar(BICMat(2:end,:),'FaceColor',0.7*ones(1,3))
defaultplot
set(gca,'XTick',1:9,'XTickLabel',modelnames(2:end));

% figure
% bar(M_BIC); hold on
% errorbar(M_BIC,SEM_BIC,'k','LineStyle','none')
% title('BIC')
% xlim([0.5 nModels+0.5])
% set(gca,'XTick',1:10,'XTickLabel',modelnames);
% defaultplot
