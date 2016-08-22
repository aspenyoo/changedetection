
%%  PLOTTING PARAMETER FITS

modelVec = 1:5;
subjids = {'1','3','4','nd','el'};
% isubj = 1;

for imodel = 1:length(modelVec);
    model = modelVec(imodel);
    for isubj = 1:length(subjids);
        subjid = subjids{isubj};
        
        filename = sprintf('paramfit_model%d_04292016.mat',model);
        load(filename)
        
        bfp = bestFitParam(isubj,:);
        plot_datafit(subjid,model,bfp)
    end
end

%% LOG LIKELIHOODS OF EACH MODEL

modelVec = [1 5 7];
nModels = length(modelVec);
% subjids = {'1_100000','2_100000','3_100000','4_100000','5_100000'};
% subjids = {'NN31','NN32','NN33','NN34','NN35'};
subjids = {'1','3','4','el','nd'};
nSubj = length(subjids);

LLMat = nan(nModels, nSubj);
k = nan(1,nModels);
for imodel = 1:nModels;
    model = modelVec(imodel);
    
%     load(sprintf('paramfit_model%d_05122016.mat',model)); % for new neural network (matched to real subj)
%     load(sprintf('paramfit_model%d_05112016.mat',model)); % for old neural network
    load(sprintf('paramfit_model%d_04292016.mat',model)); % load model parameter fits
    k(imodel) = size(bestFitParam,2); % number of parameters in model
    LLMat(imodel,:) = nLL_est; % MLE
end

experimenttype = 'Detection';
n = nan(1,nSubj);
for isubj = 1:nSubj;
    subjid = subjids{isubj};
    
    [data] = concatdata(subjid,experimenttype);
    n(isubj) = size(data,1);
end
nMat = repmat(n,nModels,1);
kMat = repmat(k',1,nSubj);

AICMat = 2.*(LLMat + kMat);
AICcMat = AICMat + (2.*kMat.*(kMat+1))./(nMat-kMat-1);
BICMat = 2.*LLMat + kMat + log(nMat);

subjidx = [1 4 3 2 5];
idx = 3;
blahidx = 1:4;
blahidx(idx) = [];

% plot AIC
figure
blah = [bsxfun(@minus,AICMat(idx,:),AICMat); 1:5];
blah = blah(blahidx(1:end-1),subjidx)';
barh(blah)
defaultplot
ax = gca;
% ax.XTickLabel = ['S1';'S2';'S3';'S4';'S5'];
title('AIC_{heurs} - AIC');

% plot AICc
figure;
blah = bsxfun(@minus,AICcMat(idx,:),AICcMat);
bar(blah')
defaultplot
set(gca,'XTickLabel',['S1';'S2';'S3';'S4';'S5']);
% ax.XTickLabel = ['S1';'S2';'S3';'S4';'S5'];
title('AICc - AICc_{heurs}');

% average AICc
figure;
aveblah = mean(blah,2);
semblah = std(blah,[],2)/sqrt(nSubj);
bar(aveblah(1:2),'w'); hold on
errorbar(aveblah(1:2),semblah(1:2),'LineStyle','none','Color','k')
defaultplot
set(gca,'XTickLabel',[{'optimal'};{'fixed'}]);
title('AICc - AICc_{heurs}');

% plot BIC
figure;
blah = bsxfun(@minus,BICMat(idx,:),BICMat);
% blah = blah(blahidx(1:end-1),subjidx)';
bar(blah')
defaultplot
ax = gca;
ax.XTickLabel = ['S1';'S2';'S3';'S4';'S5'];
title('BIC_{heurs} - BIC');

% average BIC
figure;
aveblah = mean(blah,2);
semblah = std(blah,[],2)/sqrt(nSubj);
bar(aveblah(1:2),'w'); hold on
errorbar(aveblah(1:2),semblah(1:2),'LineStyle','none','Color','k')
defaultplot
set(gca,'XTickLabel',[{'optimal'};{'fixed'}]);
title('BIC - BIC_{heurs}');

%% looking at how linear change in sigma changes the decision criteria placement

prior_mu = 0;
pcommon = 0.5;
prior_kappa = 2.6982;
nKappatilde = 10;
kappatildeVec = exp(linspace(-4.5,3,nKappatilde));

Izero = @(x_) besseli(0,x_,1);  % Rescaled Bessel function of the 1st kind

d = nan(1,nKappatilde);
for ikappa = 1:nKappatilde;
    kappatilde = kappatildeVec(ikappa);
    kappabar = sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu));
    d(ikappa) = log(pcommon/(1-pcommon)) + kappatilde ...
        +(prior_kappa-kappabar).*log(Izero(prior_kappa)./Izero(kappabar));
end

plot(kappatildeVec,abs(d),'o');
defaultplot
xlabel('kappatilde')
ylabel('decision boundary')

%% MEMORY UNCERTAINTY DIDACTIC FIGURE
% May 2nd 2016

xx = linspace(-60,60,100);

sigmaVec = [0.1 5 10];

plotcolors = [aspencolors('berry'); aspencolors('dustyrose'); aspencolors('palepink')];

figure;
for isigma = 1:length(sigmaVec);
    
    sigma = sigmaVec(isigma);
    
    nochangedist = normpdf(xx,0,sigma);%/sum(normpdf(xx,0,sigma));
    changedist = normpdf(xx,0,sigma+20);%/sum(normpdf(xx,0,sigma+20));
    
    
    plot(xx,nochangedist,'Color',plotcolors(isigma,:))
    hold on
    plot(xx,changedist,'Color',plotcolors(isigma,:))
    
end
defaultplot

%% plot empirial uncertainty decision boundary plot

modelVec = [1 5 7];
nModels = length(modelVec);
subjids = {'1','3','4','el','nd'};
nSubj = length(subjids);
nCond = 5;

Izero = @(x_) besseli(0,x_,1);  % Rescaled Bessel function of the 1st kind

boundary = nan(nSubj,nCond,nModels);
for imodel = 1:nModels;
    model = modelVec(imodel);
    
    load(sprintf('paramfit_model%d_04292016.mat',model)); % load model parameter fits
    for isubj = 1:nSubj
        prior_kappa = 2.6982;
        prior_mu = 0;
        pcommon = 0.5; % Probability of sameness P(C=1)
        switch model
            case 1 % optimal model
                % get decision boundary for each condition
                for icond = 1:nCond;
                    kappa = bestFitParam(isubj,icond);
                    kappatilde = kappa;
                    
                    %                     kappabar = (sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x)));
                    d = @(x) abs(log(pcommon/(1-pcommon)) + kappatilde.*cos(x) ...
                        +(prior_kappa-(sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x)))...
                        ).*log(Izero(prior_kappa)./Izero((sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x))))));
                    boundary(isubj,icond,imodel) = fmincon(d,rand*pi,[],[],[],[],0,pi);
                    
                end
                
            case 5 % fixed model
                % get decision boundary for one condition..same for all.
                kappa = bestFitParam(isubj,nCond+1);
                kappatilde = kappa;
                
                %                     kappabar = (sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x)));
                d = @(x) abs(log(pcommon/(1-pcommon)) + kappatilde.*cos(x) ...
                    +(prior_kappa-(sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x)))...
                    ).*log(Izero(prior_kappa)./Izero((sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - x))))));
                blah = fmincon(d,rand*pi,[],[],[],[],0,pi);
                boundary(isubj,:,imodel) = blah*ones(nCond,1);
                
            case 7 % heuristic model
                % get decision boundary for each condition.
                kcommonVec = bestFitParam(isubj,nCond+1:nCond+2);
                for icond = 1:nCond;
                    kappa = bestFitParam(isubj,icond);
                    kappa1 = bestFitParam(isubj,1);
                    boundary(isubj,icond,imodel) = (kcommonVec(2)-kcommonVec(1))*(kappa-kappa1)+kcommonVec(1);
                end
                
        end
    end
    
    clf
    plot(bestFitParam(:,1:nCond)',boundary(:,:,imodel)','.-','MarkerSize',14)
    pause;
    
end
%%

nBlah = 10;
xx = linspace(1e-3,20,nBlah);
for iblah = 1:nBlah;
    bleh = xx(iblah);
     d = @(x) abs(bleh.*cos(x) ...
         +(prior_kappa-(sqrt(bleh^2 + prior_kappa^2 + 2*bleh*prior_kappa*cos(x)))...
         ).*log(Izero(prior_kappa)./Izero((sqrt(bleh^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(x))))));
    
    boundd(iblah) = fmincon(d,rand*pi,[],[],[],[],0,pi);
end

plot(xx,boundd)

%% make emin's NN designMats the right format

subjids = {'1','3','5'};
nSubj = length(subjids);
filepath = 'VSS 2016/submission/new neural network data/repetition 1';
nIter = '200000';

for isubj = 1:nSubj;
    subjid = subjids{isubj};
    title = ['subject_' subjid '__iter_' nIter];

    
    concatfilename = [title '.mat'];
    currfilename = [filepath '/' concatfilename];
    load(currfilename);
    
    designMat(:,1) = rad2deg(designMat(:,1))./2; % delta
    designMat(:,2) = ones(size(designMat,1),1); % set size
    designMat(:,5) = 10*ones(size(designMat,1),1); % conditions

    % reliabilities
    blahs = unique(designMat(:,3));
    blah2Vec = [0.15 .3 0.5 .8 .999];
    for i = 1:length(blahs);
        idx = designMat(:,3) == blahs(i);
        designMat(idx,3) = blah2Vec(i);
    end
    
    designMat(:,6) = 1- designMat(:,6);
    save(['Exp_ChangeDetection_Simult_concat_subj' subjid '_' nIter '.mat'],'designMat');
    
end