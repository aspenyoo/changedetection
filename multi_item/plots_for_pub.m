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


%% EXP 1 MODEL FITS

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

% % calculated AIC, AICc, and BIC
% BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials)); % BIC
% BICMat = bsxfun(@minus,BICMat,BICMat(1,:)); % BIC relative to VVO
%
% % mean sem
% m = nanmean(BICMat,2);
% sem = nanstd(BICMat,[],2)/sqrt(nSubj);

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
% condition = 'combined';
% additionalpaths = '';%'ellipse_keshvari/';%'';%'combined_diffdisp/';%
load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition));
modelnames = {  'VVO', 'VFO', 'VSO',...
    'VVM', 'VFM', 'VSM',...
    'FFO', 'FSO',...
    'FFM', 'FSM'};
nModels = length(modelnames);
nTrials = 2000;

% calculated AIC, AICc, and BIC
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));
nSubj = size(BICMat,2);

BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

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

%% comparing model families

clear all
close all

condition = 'Ellipse';
additionalpaths = 'ellipse_keshvari/';%'';%'combined_diffdisp/';%
% condition = 'combined';
% additionalpaths = '';%'ellipse_keshvari/';%'';%'combined_diffdisp/';%
load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition));
modelnames = {  'VVO', 'VFO', 'VSO',...
                'VVM', 'VFM', 'VSM',...
                       'FFO', 'FSO',...
                       'FFM', 'FSM'};
nModels = length(modelnames);
nTrials = 2000;

% calculated AIC, AICc, and BIC
BICMat = 2*bsxfun(@plus,LLMat,nParamsVec' + log(nTrials));
nSubj = size(BICMat,2);

% BICMat = bsxfun(@minus,BICMat,BICMat(1,:));

% mean sem of same thing
% M_BIC = nanmean(BICMat,2);
% SEM_BIC = nanstd(BICMat,[],2)/sqrt(nSubj);

% compare V?? vs F?? models of same type
models_v = [2 3 7 9];
models_f = [5 6 9 10];

moddiff = BICMat(models_v,:) - BICMat(models_f,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_f))

% VV vs FF
models_v = [1 4];
models_f = [7 9];

moddiff = BICMat(models_v,:) - BICMat(models_f,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_f))

% comparing ?V? vs ?F? and ?S?
models_v = [1 4];
models_f = [2 5];
models_s = [3 6];

moddiff = BICMat(models_v,:) - BICMat(models_f,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_f))

moddiff = BICMat(models_v,:) - BICMat(models_s,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_f))

% comparison ?F? and ?S?
models_f = [2 5 7 8];
models_s = [3 6 9 10];

moddiff = BICMat(models_f,:) - BICMat(models_s,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_f))

% comparing ??O vs ??M
models_o = [1 2 3 7 8];
models_m = [4 5 6 9 10];

moddiff = BICMat(models_o,:) - BICMat(models_m,:);
mean(moddiff,2)
std(moddiff,[],2)./sqrt(length(models_o))

%% MAKING BIC ANOVA txt

nSubj=13;
nModels = 10;

% modelnames = {  'VVO', 'VFO', 'VSO',...
%     'VVM', 'VFM', 'VSM',...
%     'FFO', 'FSO',...
%     'FFM', 'FSM'};
encoding = repmat([ones(6,1); 2*ones(4,1)],1,nSubj);
decoding = repmat([1;2;3;1;2;3;2;3;2;3],1,nSubj);
drule = repmat([1;1;1;2;2;2;1;1;2;2],1,nSubj);
subj = repmat(1:13,nModels,1);

Subject = subj(:);
Encoding = encoding(:);
Decoding = decoding(:);
Decision = drule(:);
BIC = BICMat(:);

t = table(Subject, Encoding, Decoding, Decision, BIC);
writetable(t,'exp1_BIC.txt');

%% EXP 2 MODEL FITS

clear all
% condition = 'Ellipse';
% additionalpaths = 'ellipse_keshvari/';
condition = 'combined';
additionalpaths = '';%'combined_diffdisp/';%'ellipse_keshvari/';
additionalpaths2 = '';%'_keshvari';

modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
    1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
    2 2 1; 2 3 1; ...  % F_O model variants
    2 2 2; 2 3 2];     % F_M model variants
imodelVec = [1 2 3 7 8];
% imodelVec = [4 5 6 9 10];

load(sprintf('analysis/fits/%sbfp_%s.mat',additionalpaths,condition));
modelnames = {  'VVO', 'VFO', 'VSO',...
    'VVM', 'VFM', 'VSM',...
    'FFO', 'FSO',...
    'FFM', 'FSM'};
nSubj = size(LLMat,2);
nTrials = 2000;

conditionVec = {'Ellipse','Line'};
nConditions = 2;

% prediction stuff
nSamples = 50;
nBins = 6;
quantilebinning=1;

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),5));
colorMat = cmap(idxs,:);

nModels = length(imodelVec);
[x_mean_e, pc_data_e, pc_pred_e, x_mean_l, pc_data_l, pc_pred_l] = deal(nan(5,nBins,nSubj));
[HRallVec_e,HRlowVec_e,HRhighVec_e,FARVec_e,mod_HRallVec_e,mod_HRlowVec_e,...
    mod_HRhighVec_e,mod_FARVec_e,HRallVec_l,HRlowVec_l,HRhighVec_l,...
    FARVec_l,mod_HRallVec_l,mod_HRlowVec_l,mod_HRhighVec_l,mod_FARVec_l] = deal(nan(nSubj,5));
for imodel = 1:nModels;%imodelVec%1:2;%nModels
    modelnum = imodelVec(imodel);
    model = modelMat(modelnum,:);
    
    % load bfp fits
    %     load(sprintf('analysis/fits/%sbfp_%s%s.mat',additionalpaths,condition,additionalpaths2))
    bfpmat = bfpMat{modelnum};
    
    for isubj = 1:nSubj
        subjid = subjidVec{isubj};
        bfp = bfpmat(isubj,:);
        
        % load data
        load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid),'data');
        data_E = data;
        load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid),'data');
        data_L = data;
        
        % get predictions
        [LL,p_C_hat] = calculate_joint_LL(bfp,data_E,data_L,model,[],nSamples);
        fprintf('subj %s: %5.2f \n',subjid,LL)
        
        figure(1);
        [x_mean_e(:,:,isubj), pc_data_e(:,:,isubj), pc_pred_e(:,:,isubj)] = plot_psychometric_fn(data_E,nBins,p_C_hat.Ellipse,quantilebinning);
        [x_mean_l(:,:,isubj), pc_data_l(:,:,isubj), pc_pred_l(:,:,isubj)] = plot_psychometric_fn(data_L,nBins,p_C_hat.Line,quantilebinning);
        
        % get hits/false alarms
        [HRallVec_e(isubj,:),HRlowVec_e(isubj,:),HRhighVec_e(isubj,:),FARVec_e(isubj,:),...
            mod_HRallVec_e(isubj,:),mod_HRlowVec_e(isubj,:),mod_HRhighVec_e(isubj,:),mod_FARVec_e(isubj,:)] = ...
            plot_HR_FAR(data_E,p_C_hat.Ellipse,0);
        [HRallVec_l(isubj,:),HRlowVec_l(isubj,:),HRhighVec_l(isubj,:),FARVec_l(isubj,:),...
            mod_HRallVec_l(isubj,:),mod_HRlowVec_l(isubj,:),mod_HRhighVec_l(isubj,:),mod_FARVec_l(isubj,:)] = ...
            plot_HR_FAR(data_L,p_C_hat.Line,0);
        
        %     for icondition = 1:nConditions;
        %         condition = conditionVec{icondition};
        %         figure(1);
        %         [x_mean, pc_data, pc_pred] = deal(nan(5,nBins,nSubj));
        %         [HRallVec,HRlowVec,HRhighVec,FARVec,mod_HRallVec,mod_HRlowVec,mod_HRhighVec,mod_FARVec] = deal(nan(nSubj,5));
        %         for isubj = 1:nSubj
        %             subjid = subjidVec{isubj};
        %             bfp = bfpmat(isubj,:);
        %
        %             % load data
        %             load(sprintf('data/fitting_data/%s_%s_simple.mat',subjid,condition),'data')
        %
        %             % get predictions
        %             [LL,p_C_hat] = calculate_LL(bfp,data,model,[],nSamples);
        %             fprintf('subj %s: %5.2f \n',subjid,LL)
        %
        %             % get psychometric function binned data
        %             [x_mean(:,:,isubj), pc_data(:,:,isubj), pc_pred(:,:,isubj)] = plot_psychometric_fn(data,nBins,p_C_hat.(condition),quantilebinning);
        %
        %             % get hits/false alarms binned data
        %             [HRallVec(isubj,:),HRlowVec(isubj,:),HRhighVec(isubj,:),FARVec(isubj,:),...
        %                 mod_HRallVec(isubj,:),mod_HRlowVec(isubj,:),mod_HRhighVec(isubj,:),mod_FARVec(isubj,:)] = ...
        %                 plot_HR_FAR(data,p_C_hat.(condition),0);
        
    end
    
    % get participant and model means: psychometric fn
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
    
    % get participant and model means: hits/false alarms
    m_HRall_e = mean(HRallVec_e);
    m_HRlow_e = mean(HRlowVec_e);
    m_HRhigh_e = mean(HRhighVec_e);
    m_FAR_e = mean(FARVec_e);
    sem_HRall_e = std(HRallVec_e)./sqrt(nSubj);
    sem_HRlow_e = std(HRlowVec_e)./sqrt(nSubj);
    sem_HRhigh_e = std(HRhighVec_e)./sqrt(nSubj);
    sem_FAR_e = std(FARVec_e)./sqrt(nSubj);
    m_mod_HRall_e = mean(mod_HRallVec_e);
    m_mod_HRlow_e = mean(mod_HRlowVec_e);
    m_mod_HRhigh_e = mean(mod_HRhighVec_e);
    m_mod_FAR_e = mean(mod_FARVec_e);
    sem_mod_HRall_e = std(mod_HRallVec_e)./sqrt(nSubj);
    sem_mod_HRlow_e = std(mod_HRlowVec_e)./sqrt(nSubj);
    sem_mod_HRhigh_e = std(mod_HRhighVec_e)./sqrt(nSubj);
    sem_mod_FAR_e = std(mod_FARVec_e)./sqrt(nSubj);
    
    m_HRall_l = mean(HRallVec_l);
    m_HRlow_l = mean(HRlowVec_l);
    m_HRhigh_l = mean(HRhighVec_l);
    m_FAR_l = mean(FARVec_l);
    sem_HRall_l = std(HRallVec_l)./sqrt(nSubj);
    sem_HRlow_l = std(HRlowVec_l)./sqrt(nSubj);
    sem_HRhigh_l = std(HRhighVec_l)./sqrt(nSubj);
    sem_FAR_l = std(FARVec_l)./sqrt(nSubj);
    m_mod_HRall_l = mean(mod_HRallVec_l);
    m_mod_HRlow_l = mean(mod_HRlowVec_l);
    m_mod_HRhigh_l = mean(mod_HRhighVec_l);
    m_mod_FAR_l = mean(mod_FARVec_l);
    sem_mod_HRall_l = std(mod_HRallVec_l)./sqrt(nSubj);
    sem_mod_HRlow_l = std(mod_HRlowVec_l)./sqrt(nSubj);
    sem_mod_HRhigh_l = std(mod_HRhighVec_l)./sqrt(nSubj);
    sem_mod_FAR_l = std(mod_FARVec_l)./sqrt(nSubj);
    %         save('subjpsychometricfn.mat','xrange','partM','partSEM',...
    %             'm_HRall','m_HRlow','m_HRhigh','m_FAR','sem_HRall','sem_HRlow',...
    %             'sem_HRhigh','sem_FAR')
    
    % =================== PLOT =======================
    figure(2);
    
    % PLOT MODEL FITS
    subplot(nModels,4,(imodel-1)*4+1); hold on
    xlim([-0.2 pi/2+0.2])
    ylim([0 1])
    for ii = 1:5;
        plot_summaryfit(xrange_e(ii,:),partM_e(ii,:),partSEM_e(ii,:),modelM_e(ii,:),...
            modelSEM_e(ii,:),colorMat(ii,:),colorMat(ii,:))
    end
    set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
    
    % PLOT MODEL FITS
    subplot(nModels,4,(imodel-1)*4+2); hold on
    xlim([-0.2 pi/2+0.2])
    ylim([0 1])
    for ii = 1:5;
        plot_summaryfit(xrange_l(ii,:),partM_l(ii,:),partSEM_l(ii,:),modelM_l(ii,:),...
            modelSEM_l(ii,:),colorMat(ii,:),colorMat(ii,:))
    end
    set(gca,'XTick',[0:.25:1].*pi/2,'XTickLabel',[0:.25:1].*pi/2)
    
    % PLOT HITS FALSE ALARMS
    subplot(nModels,4,(imodel-1)*4+3); hold on;
    xlim([-0.5 4.5])
    ylim([0 1])
    plot_summaryfit(0:4,m_HRall_e,sem_HRall_e,m_mod_HRall_e,sem_mod_HRall_e,colorMat(1,:),colorMat(1,:));
    plot_summaryfit(0:4,m_HRlow_e,sem_HRlow_e,m_mod_HRlow_e,sem_mod_HRlow_e,colorMat(2,:),colorMat(2,:));
    plot_summaryfit(0:4,m_HRhigh_e,sem_HRhigh_e,m_mod_HRhigh_e,sem_mod_HRhigh_e,colorMat(3,:),colorMat(3,:));
    plot_summaryfit(0:4,m_FAR_e,sem_FAR_e,m_mod_FAR_e,sem_mod_FAR_e,colorMat(4,:),colorMat(4,:));
    set(gca,'XTick',0:4,'XTickLabel',0:4)
    
    % PLOT HITS FALSE ALARMS
    subplot(nModels,4,(imodel-1)*4+4); hold on;
    xlim([-0.5 4.5])
    ylim([0 1])
    plot_summaryfit(0:4,m_HRall_l,sem_HRall_l,m_mod_HRall_l,sem_mod_HRall_l,colorMat(1,:),colorMat(1,:));
    plot_summaryfit(0:4,m_HRlow_l,sem_HRlow_l,m_mod_HRlow_l,sem_mod_HRlow_l,colorMat(2,:),colorMat(2,:));
    plot_summaryfit(0:4,m_HRhigh_l,sem_HRhigh_l,m_mod_HRhigh_l,sem_mod_HRhigh_l,colorMat(3,:),colorMat(3,:));
    plot_summaryfit(0:4,m_FAR_l,sem_FAR_l,m_mod_FAR_l,sem_mod_FAR_l,colorMat(4,:),colorMat(4,:));
    set(gca,'XTick',0:4,'XTickLabel',0:4)

end

% end
