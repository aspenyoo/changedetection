% plot real data with fake data
function plot_datafit(subjidVec, model, thetaVec)
if iscell(subjidVec); summaryplot = 1; nSubj = length(subjidVec);
else summaryplot = 0; nSubj = 1; end

% nLevels = 9;
nCond = 5;
stimLevels = cell(1,nCond); trialNums = stimLevels; nResps = stimLevels;
p_resp = cell(1,nCond);
for isubj = 1:nSubj;
    
    try
        subjid = subjidVec{isubj};
    catch
        subjid = subjidVec;
    end
    theta = thetaVec(isubj,:);
    
    if strcmp(subjid(1),'F'); % if fake data
        load(['fakedata_subj' subjid '.mat'],'Xdet');
    else
        % load real data
        experimenttype = 'Detection';
        [data] = concatdata(subjid,experimenttype);
        [Xdet] = conditionSeparator(data);
    end
    %     nCond = length(Xdet);
    
    if (summaryplot)
        [stimlevels, trialnums, nresps] = conditionSeparator(Xdet);
        stimLevels = cellfun(@(x,y) [x;y],stimLevels,stimlevels,'UniformOutput',false);
        trialNums = cellfun(@(x,y) [x;y],trialNums,trialnums,'UniformOutput',false);
        nResps = cellfun(@(x,y) [x;y],nResps,nresps,'UniformOutput',false);
    else
        % plot real data
%         figure;
        plotrealdata({subjid}, 'Detection')
%         plotdata(Xdet);
    end
    
    % adjusting theta to match log stuff
    switch model
        case 1 % Optimal model, only free noise and lapse
            logflag = logical([ones(1,nCond) 0]);
        case 2 % Optimal model + free prior on (C=1)
            logflag = logical([ones(1,nCond) 0 0]);
        case 3 % Optimal model + free prior on C + free prior on S
            logflag = logical([ones(1,nCond) 0 1 0]);
        case 4 % Optimal model + free prior on S
            logflag = logical([ones(1,nCond+1) 0]);
        case 5 % Fixed criterion model: free noise, sigmatilde, lapse
            logflag = logical([ones(1,nCond+1) 0]);
        case 6 % Super-free model: free noise and full free sigmatilde and lapse
            logflag = logical([ones(1,2*nCond) 0]);
        case 7 % linear heuristic model: free noise, low and high kcommon, lapse
            logflag = logical([ones(1,nCond) zeros(1,3)]);
    end
    theta(logflag) = log(theta(logflag));
    
    % plot predicted data (given parameters and model)
    plotcolors = aspencolors(nCond,'blue');
    nSamps = 100;
    xx = linspace(-90,90,nSamps)';
    fakedata = cell(1,nCond);
    fakedata = cellfun(@(x) [xx ones(size(xx))],fakedata,'UniformOutput',false);
    [~,presp] = loglike(fakedata, model, theta);
    if (summaryplot)
        p_resp = cellfun(@(x,y) [x;y'],p_resp,presp,'UniformOutput',false);
    else
        for icond = 1:nCond;
            hold on;
            plot(xx,presp{icond},'Color',plotcolors(icond,:),'LineWidth',2)
        end
    end
end

% plotting summary plot
if (summaryplot)
    % checking if stimLevels are the same
    stimLevels = stimLevels{1}(1,:); % all stim levels should be the same...
    data_pResps = cellfun(@(x,y) x./y,nResps,trialNums,'UniformOutput',false);
    
    % mean and std of data per condition
    mean_data = cellfun(@(x) mean(x),data_pResps,'UniformOutput',false);
    sem_data = cellfun(@(x) std(x)/sqrt(nSubj),data_pResps,'UniformOutput',false);
    
    % mean and std of model
    mean_model = cellfun(@(x) mean(x),p_resp,'UniformOutput',false);
    sem_model = cellfun(@(x) std(x)/sqrt(nSubj),p_resp,'UniformOutput',false);

    % plot!    
    condVec = [1:5];
    for cond = 1:length(condVec);
        icond = condVec(cond);
        
        plot_summaryfit(stimLevels,mean_data{icond},sem_data{icond},[],[],plotcolors(icond,:),plotcolors(icond,:));
        plot_summaryfit(xx',[],[],mean_model{icond},sem_model{icond},plotcolors(icond,:),plotcolors(icond,:))
    end
    axis([-45 45 0 1])
%     xlim([-45 45])
    ax = gca;
    ax.XTick = [-40 -20 0 20 40];
    ax.YTick = [0 0.5 1];
    
end