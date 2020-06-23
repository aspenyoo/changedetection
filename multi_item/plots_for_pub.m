%% exp criterion

% clear all
%
% condition = 'combined';
% subjidVec = {'S02','S03','S06','S08','S10','S11','S14',...
%     'S15','S16','S17','S19','S20','S23'};
% nSubj = length(subjidVec);
%
% modelVec = [1 3 2; 2 3 2];
% nModels = size(modelVec,1);
%
% for imodel = 1:nModels;
%     model = modelVec(imodel,:);
%
%     for isubj = 1:nSubj;
%         subjid = subjidVec{isubj};
%
%         load(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,...
%             condition,model(1),model(2),model(3)))
%
%         bfp(:,end) = exp(bfp(:,end));
%
%         save(sprintf('analysis/fits/subj%s_%s_model%d%d%d.mat',subjid,...
%             condition,model(1),model(2),model(3)),'bfp','completedruns','LLVec')
%     end
% end

%% ====================================================================
%                 PLOTS OF RAW DATA (NO MODEL FITS)
% ====================================================================

%% produces figure with four following plots:
% - all proportion report change (p("change")) for all trials
% - hits: p("change") for trials with low rel ellipse change
% - hits: p("change") for trials with high rel ellipse change
% - hits/FAs: as a function of number of high-rel ellipses

clear all
condition = 'Ellipse';
noisecond = 3; % 1: no, 2: local, 3: global decision noise

load('modelfittingsettings.mat')
modelnamesVec = modelnamesVec((-13:0)+(noisecond*14));
modelMat = modelMat((-13:0)+(noisecond*14),:);

subjidVec = {'S02','S03','S06','S08','S10','S11','S14',...
    'S15','S16','S17','S19','S20','S23'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; 1 4 1; ...  % V_O model variants
    1 1 2;  1 2 2; 1 3 2; 1 4 2; ...  % V_M model variants
    2 2 1; 2 3 1; 2 4 1; ...  % F_O model variants
    2 2 2; 2 3 2; 2 4 2];     % F_M model variants

nSubj = length(subjidVec);
nModels = size(modelMat,1);

nSamples = 1000;
nBins = 8;
quantilebinedges = 11;

colorMat1 = ([223 171 110; ...
    212  95 100; ...
    169 109 223; ...
    126 159 235; ...
    117 214 180])./255;

% ====== get all data used for plotting ======
figure;
set(gcf,'Position',[0 600 1000 200])
[x_mean, pc_data] = deal(nan(5,nBins,nSubj));
[HRallVec,HRlowVec,HRhighVec,FARVec] = deal(nan(nSubj,5));
[s_x_mean, s_pc_data] = deal(nan(5,nBins,2,nSubj));
for isubj = 1:nSubj
    subjid = subjidVec{isubj};
    
    % load data
    load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')
    
    % psychometric fn
    [x_mean(:,:,isubj), pc_data(:,:,isubj)] = plot_psychometric_fn(data,nBins,[],quantilebinedges);
    
    % split psychometric fn
    [s_x_mean(:,:,:,isubj), s_pc_data(:,:,:,isubj)] = plot_psychometric_fn_split(data,nBins,[],quantilebinedges);
    
    % get hits false alarms
    [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:)] = ...
        plot_HR_FAR(data,[],0);
end

% ===== psychometric fn =======

% get participant and model means
xrange = mean(x_mean,3);
partM = mean(pc_data,3);
partSEM = std(pc_data,[],3)./sqrt(nSubj);
mult = 0.05; % errorbar multiplier

subplot(1,4,1)
for ii = 1:5;
    hold on
    h = errorbar(xrange(ii,:),partM(ii,:),partSEM(ii,:),...
        'color',colorMat1(ii,:));
    try
        errbar(h,mult);
    end
    % plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),[],...
    %     [],colorMat(ii,:),[])
end
defaultplot
axis([-0.1 pi/2+0.1 0 1])
set(gca,'XTick',0:pi/8:pi/2,'XTickLabel',{0,'','\pi/4','','\pi/2'},...
    'YTick',0:0.1:1,'YTickLabel',{0, '','','','',0.5,'','','','',1})
xlabel('amount of change')
ylabel('proportion report change')


% ======= split psychometric function  ========

for irel = 1:2;
    % get participant and model means
    xrange = mean(s_x_mean(:,:,irel,:),4);
    partM = mean(s_pc_data(:,:,irel,:),4);
    partSEM = std(s_pc_data(:,:,irel,:),[],4)./sqrt(nSubj);
    
    %     % get colormap info
    %     h = figure(99);
    %     cmap = colormap('parula'); % get a rough colormap
    %     close(h)
    %     idxs = round(linspace(1,size(cmap,1),5));
    %     colorMat = cmap(idxs,:);
    
    subplot(1,4,irel+1)
    hold on;
    for ii = 1:5;
        h = errorbar(xrange(ii,:),partM(ii,:),partSEM(ii,:),...
            'color',colorMat1(ii,:));
        try
            errbar(h,mult);
        end
    end
    axis([-0.1 pi/2+0.1 0 1])
    set(gca,'XTick',0:pi/8:pi/2,'XTickLabel',{0,'','\pi/4','','\pi/2'},...
        'YTick',0:0.1:1,'YTickLabel',{})
    xlabel('amount of change')
    defaultplot
end

% ======= hits false alarms ========

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
subplot(1,4,4); hold on
errorbar(0:4,m_HRall,sem_HRall)
errorbar(0:4,m_HRlow,sem_HRlow)
errorbar(0:4,m_HRhigh,sem_HRhigh)
errorbar(0:4,m_FAR,sem_FAR)
axis([-0.5 4.5 0 1])
set(gca,'Xtick',0:4,'YTick',0:0.1:1,'YTickLabel',{})
xlabel('number of high-rel stim')
defaultplot


%% plot raw data for both experiments

% this cell produces a figure with four subplots:
% - p("change") for the Ellipse experiment, broken up by number of high-rel ellipses (nHigh)
% - p("change") for Line exp, broken up by nHigh
% - Ellipse exp: hits and false alarms as a function of nHigh
% - Line exp: hits and false alarms as a function of nHigh

% notes to self:
% - subjpsychometricfn seems to be resaved and is a vague title. pls fix!

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
colorMat1 = cmap(idxs,:);

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

pc_e = pc_data;
hrall_e = HRallVec;
hrlow_e = HRlowVec;
hrhigh_e = HRhighVec;
far_e = FARVec;

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
    plot(xrange(ii,:),partM(ii,:),'Color',colorMat1(ii,:))
    errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat1(ii,:))
end
set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
defaultplot
xlabel('magnitude change')
%     ylabel('proportion respond change')

% PLOT HITS FALSE ALARMS
subplot(1,4,3), hold on;
xlim([-0.5 4.5])
ylim([0 1])
plot(0:4,m_HRall,'Color',colorMat1(1,:))
plot(0:4,m_HRlow,'Color',colorMat1(2,:))
plot(0:4,m_HRhigh,'Color',colorMat1(3,:))
plot(0:4,m_FAR,'Color',colorMat1(4,:))
errorb(0:4,m_HRall,sem_HRall,'color',colorMat1(1,:))
errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat1(2,:))
errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat1(3,:))
errorb(0:4,m_FAR,sem_FAR,'color',colorMat1(4,:))
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

pc_l = pc_data;
hrall_l = HRallVec;
hrlow_l = HRlowVec;
hrhigh_l = HRhighVec;
far_l = FARVec;

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
    plot(xrange(ii,:),partM(ii,:),'Color',colorMat1(ii,:))
    errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat1(ii,:))
end
set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
defaultplot
xlabel('magnitude change')
%     ylabel('proportion respond change')

% PLOT HITS FALSE ALARMS
subplot(1,4,4), hold on;
xlim([-0.5 4.5])
ylim([0 1])
plot(0:4,m_HRall,'Color',colorMat1(1,:))
plot(0:4,m_HRlow,'Color',colorMat1(2,:))
plot(0:4,m_HRhigh,'Color',colorMat1(3,:))
plot(0:4,m_FAR,'Color',colorMat1(4,:))
errorb(0:4,m_HRall,sem_HRall,'color',colorMat1(1,:))
errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat1(2,:))
errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat1(3,:))
errorb(0:4,m_FAR,sem_FAR,'color',colorMat1(4,:))
set(gca,'XTick',0:4,'XTickLabel',0:4)
defaultplot
xlabel('number of high reliability ellipses')
%     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')


%% DATA alone plot
% 01/29/2020: this plot seems to be plotting the above, but for one
% condition only. doesn't work

% clear all
%
% load('subjpsychometricfn.mat')
%
% h = figure(99);
% cmap = colormap('parula'); % get a rough colormap
% close(h)
% idxs = round(linspace(1,size(cmap,1),5));
% colorMat = cmap(idxs,:);
%
% % PLOT MODEL FITS
% subplot(1,2,1), hold on;
% xlim([-0.2 pi/2+0.2])
% ylim([0 1])
% for ii = 1:5;
%     errorb(xrange(ii,:),partM(ii,:),partSEM(ii,:),'color',colorMat(ii,:))
% end
% set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
% defaultplot
% xlabel('magnitude change')
% %     ylabel('proportion respond change')
%
% % PLOT HITS FALSE ALARMS
% subplot(1,2,2), hold on;
% xlim([-0.5 4.5])
% ylim([0 1])
% errorb(0:4,m_HRall,sem_HRall,'color',colorMat(1,:))
% errorb(0:4,m_HRlow,sem_HRlow,'color',colorMat(2,:))
% errorb(0:4,m_HRhigh,sem_HRhigh,'color',colorMat(3,:))
% errorb(0:4,m_FAR,sem_FAR,'color',colorMat(4,:))
% set(gca,'XTick',0:4,'XTickLabel',0:4)
% defaultplot
% xlabel('number of high reliability ellipses')
% %     legend('hits: all', 'hits: low rel','hits: high rel','false alarms')
%

%% ====================================================================
%                      MODEL FITS: ONE CONDITION
% ====================================================================

%% FIGURE 4: FACTORIAL MODEL FITS

clear all
condition = 'Line';

figure(2); clf;
modelcolMat = [1; ...
               5 ]';

load(sprintf('analysis/fits/bfp_%s.mat',condition));
load('modelfittingsettings.mat')

nTrials = 2000;

% calculated AIC, AICc, and BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
AICcMat = bsxfun(@minus,AICcMat,AICcMat(1,:));

% median
med_AICc = median(AICcMat,2);

% get 95 CI
nModels = size(modelMat,1);
CI_AICc= nan(2,nModels);
for imodel = 2:nModels
    AICcVec = AICcMat(imodel,:);
    
    blah = sort(median(AICcVec(randi(nSubjs,1000,nSubjs)),2));
    CI_AICc(:,imodel) = blah([25 975]);
end

nModels = size(modelcolMat,1);
for icol = 1:2
    imodelVec = modelcolMat(:,icol);
    
    for imodel = 1:nModels;%imodelVec%1:2;%nModels
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
        
        % colormap info 
        colorMat1 = ... % for subplot1
           [223 171 110;...
            212  95 100;...
            169 109 223;...
            126 159 235;...
            117 214 180]./255;
        
        colorMat2 = ... % colormap for subplot 2
           [ 99 185 137;...     % high-rel hit
            102  91 189;...     % hit
            102 161 212;...     % low-rel hit
            223 189 104]./255;  % false alarm
        
        
        % plot
        figure(2);
        
        % PLOT MODEL FITS
        idx = sub2ind([6,nModels],3*icol,imodel);
        subplot(nModels,6,idx-2); hold on
        xlim([-0.2 pi/2+0.2])
        for ii = 1:5;
            plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
                modelSEM(ii,:),colorMat1(ii,:),colorMat1(ii,:))
        end
        set(gca,'Xlim',[-pi/16 9*pi/16],'Ylim',[0 1],...
            'XTick',[0:.25:1].*pi/2,'YTick',0:.2:1,...
            'XTickLabel','','YTickLabel','');
        if (imodel==1) && (icol==1); set(gca,'YTickLabel',{0,'','','','',1}); end
        
        % PLOT HITS FALSE ALARMS
        subplot(nModels,6,idx-1); hold on;
        xlim([-0.5 4.5])
        ylim([0 1])
        plot_summaryfit(0:4,m_HRall,sem_HRall,m_mod_HRall,sem_mod_HRall,colorMat2(2,:),colorMat2(2,:));
        plot_summaryfit(0:4,m_HRlow,sem_HRlow,m_mod_HRlow,sem_mod_HRlow,colorMat2(3,:),colorMat2(3,:));
        plot_summaryfit(0:4,m_HRhigh,sem_HRhigh,m_mod_HRhigh,sem_mod_HRhigh,colorMat2(1,:),colorMat2(1,:));
        plot_summaryfit(0:4,m_FAR,sem_FAR,m_mod_FAR,sem_mod_FAR,colorMat2(4,:),colorMat2(4,:));
        set(gca,'XTick',0:4,'YTick',0:.2:1,'XTickLabel','','YTickLabel','');% 0:4)

        
        % \Delta AICc
        subplot(nModels,6,idx); hold on
        fill([0 0 nSubj+1 nSubj+1],...
            [CI_AICc(:,imodel)' CI_AICc(2,imodel) CI_AICc(1,imodel)],0.8*ones(1,3));
        bar(AICcMat(modelnum,:),'FaceColor',[234 191 51]./255,'EdgeColor','none','LineWidth',2)
        set(gca,'Xlim',[0 nSubj],'Ylim',[-100 1000],...
                'XTick',[],'XTickLabel',[],...
                'YTick', [-100 0:200:1000],'YTickLabel', '');
        if (imodel==1) && (icol==1); 
            set(gca,'YTickLabel',{'',0,'','','','',1000}); 
            ylabel('\Delta AICc')
        end
        defaultplot
    end
end

%% MODEL FITS
% 1x4 figure with the following subplots
% - full psychometric function
% - psychometric fn, for low-rel ellipse trials
% - psychometric fn, for high-rel ellipse trials
% - hits and false alarms

clear all
condition = 'Line';
noisecond = 3; % 1: no, 2: local, 3: global decision noise

load('modelfittingsettings.mat')
modelnamesVec = modelnamesVec((-13:0)+(noisecond*14));
modelMat = modelMat((-13:0)+(noisecond*14),:);

nSamples = 1000;
nBins = 8;
quantilebinedges = 11;

colorMat1 = ([223 171 110; ...
    212  95 100; ...
    169 109 223; ...
    126 159 235; ...
    117 214 180])./255;

imodel = 1;
model = modelMat(imodel,:);

% ===================== get model predictions ==============

% load bfp fits
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition))
bfpMat = bfpMat{imodel};
nSubj = length(subjidVec);

[x_mean, pc_data, pc_pred] = deal(nan(5,nBins,nSubj));
[s_x_mean, s_pc_data, s_pc_pred] = deal(nan(5,nBins,2,nSubj));
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
    [x_mean(:,:,isubj), pc_data(:,:,isubj), pc_pred(:,:,isubj)] = plot_psychometric_fn(data,nBins,p_C_hat,quantilebinedges);
    
    % get split psychometric function binned data
    [s_x_mean(:,:,:,isubj), s_pc_data(:,:,:,isubj), s_pc_pred(:,:,:,isubj)] = plot_psychometric_fn_split(data,nBins,p_C_hat,quantilebinedges);
    
    % get hits/false alarms binned data
    [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:),...
        mod_HRallVec(isubj,:),mod_HRlowVec(isubj,:),mod_HRhighVec(isubj,:),mod_FARVec(isubj,:)] = ...
        plot_HR_FAR(data,p_C_hat,0);
    
end

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
colorMat1 = cmap(idxs,:);

% ===================== plot =======================

figure;
set(gcf,'Position',[0 600 1000 200])
mult = 0.05;

% ================= psychometric fn ===================

% get participant and model means: psychometric fn
xrange = nanmean(x_mean,3);
partM = nanmean(pc_data,3);
partSEM = nanstd(pc_data,[],3)./sqrt(nSubj-1);
modelM = nanmean(pc_pred,3);
modelSEM = nanstd(pc_pred,[],3)./sqrt(nSubj-1);

subplot(1,4,1); hold on
for ii = 1:5;
    plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
        modelSEM(ii,:),colorMat1(ii,:),colorMat1(ii,:),mult)
end
xlabel('amount of change')
ylabel('proportion respond change')
set(gca,'XTick',0:pi/8:pi/2,'XTickLabel',{0,'','\pi/4','','\pi/2'},...
    'YTick',0:0.1:1)
axis([-0.1 pi/2+.1, 0, 1])

% =============== split psychometric fns =========

for irel = 1:2;
    % get participant and model means
    xrange = mean(s_x_mean(:,:,irel,:),4);
    partM = mean(s_pc_data(:,:,irel,:),4);
    partSEM = std(s_pc_data(:,:,irel,:),[],4)./sqrt(nSubj);
    modelM = nanmean(s_pc_pred(:,:,irel,:),4);
    modelSEM = nanstd(s_pc_pred(:,:,irel,:),[],4)./sqrt(nSubj-1);
    
    subplot(1,4,irel+1)
    hold on;
    for ii = 1:5;
        plot_summaryfit(xrange(ii,:),partM(ii,:),partSEM(ii,:),modelM(ii,:),...
            modelSEM(ii,:),colorMat1(ii,:),colorMat1(ii,:),mult)
    end
    axis([-0.1 pi/2+.1 0 1])
    xlabel('amount of change')
    set(gca,'XTick',0:pi/8:pi/2,'XTickLabel',{0,'','\pi/4','','\pi/2'},...
        'YTick',0:0.1:1,'YTickLabel',{})
end


% =================== hits false alarms ========
subplot(1,4,4); hold on;
plot_summaryfit(0:4,m_HRall,sem_HRall,m_mod_HRall,sem_mod_HRall,colorMat1(1,:),colorMat1(1,:));
plot_summaryfit(0:4,m_HRlow,sem_HRlow,m_mod_HRlow,sem_mod_HRlow,colorMat1(2,:),colorMat1(2,:));
plot_summaryfit(0:4,m_HRhigh,sem_HRhigh,m_mod_HRhigh,sem_mod_HRhigh,colorMat1(3,:),colorMat1(3,:));
plot_summaryfit(0:4,m_FAR,sem_FAR,m_mod_FAR,sem_mod_FAR,colorMat1(4,:),colorMat1(4,:));
xlabel('number of high-rel stim')
ylabel('proportion respond change')
% legend('hits: all', 'hits: low rel','hits: high rel','false alarms')
set(gca,'XTick',0:4)
set(gca,'YTick',0:0.1:1,'YTickLabel',{0, '','','','',0.5,'','','','',1})
xlim([-0.5 4.5])
ylim([0 1])

%% ====================================================================
%                      MODEL COMPARISON
% ====================================================================


%% bar plot model comparison (BIC)

clear all
close all

condition = 'Line';
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition));

noisecond = 3; % 1: no, 2: local, 3: global decision noise

load('modelfittingsettings.mat')
modelnamesVec = modelnamesVec((-13:0)+(noisecond*14));
modelMat = modelMat((-13:0)+(noisecond*14),:);
nModels = length(modelnamesVec);

nTrials = 2000;

% calculated BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
% AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
BICMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
% BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));
nSubj = size(BICMat,2);


BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

% mean sem of same thing
M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

figure; hold on
for imodel = 1:nModels-1
    fill([-0.5 0.5 0.5 -0.5]+imodel,[-SEM_BIC(imodel+1) -SEM_BIC(imodel+1) ...
        SEM_BIC(imodel+1) SEM_BIC(imodel+1)]+M_BIC(imodel+1),0.7*ones(1,3),'LineStyle','none')
end
bar(BICMat(2:end,:),'FaceColor',0.7*ones(1,3))
defaultplot
set(gca,'XTick',1:nModels-1,'XTickLabel',modelnamesVec(2:end));
xlabel('model')
ylabel(sprintf('%s better model fit',modelnamesVec{1}))

%% scatter plot model comparison (AICc and BIC)

clear all

condition = 'Line';
load(sprintf('analysis/fits/%s/bfp_%s.mat',condition,condition));

noisecond = 3; % 1: no, 2: local, 3: global decision noise

load('modelfittingsettings.mat')
modelnamesVec = modelnamesVec((-13:0)+(noisecond*14));
modelMat = modelMat((-13:0)+(noisecond*14),:);
nModels = length(modelnamesVec);

nTrials = 2000;

% calculated AIC, AICc, and BIC
AICMat = 2*bsxfun(@plus,LLMat,nParamsVec');
AICcMat = bsxfun(@plus,AICMat,((2.*nParamsVec.*(nParamsVec+1))./(nTrials-nParamsVec-1))');
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));
nSubj = size(BICMat,2);

BICMat = bsxfun(@minus,BICMat,BICMat(1,:));
AICcMat = bsxfun(@minus,AICcMat,AICcMat(1,:));

% mean sem of same thing
M_AICc = nanmean(AICcMat,2);
SEM_AICc = nanstd(AICcMat,[],2)/sqrt(nSubj);
M_BIC = nanmean(BICMat,2);
SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

figure; hold on
for imodel = 1:nModels-1
    fill([-0.5 0.5 0.5 -0.5]+imodel,[-SEM_BIC(imodel+1) -SEM_BIC(imodel+1) ...
        SEM_BIC(imodel+1) SEM_BIC(imodel+1)]+M_BIC(imodel+1),0.7*ones(1,3),'LineStyle','none')
end
x = repmat((1:nModels-1)',1,nSubj);
y = BICMat(2:end,:);
scatter(x(:),y(:))
defaultplot
set(gca,'XTick',1:nModels-1,'XTickLabel',modelnamesVec(2:end));
xlabel('model')
ylabel(sprintf('%s better model fit',modelnamesVec{1}))
