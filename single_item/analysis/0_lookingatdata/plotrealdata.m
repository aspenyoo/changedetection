function [pc] = plotrealdata(subjids, experimenttype, conditions)
% plotdata(subjids,experimenttype) gives rudementary psychometric plots of
% data to see how subjects are doing
% 
% this is useful to check that subjects are doing okay...that they are
% worth running for 4 sessions
%
% ========== INPUT VARIABLES ==========
% SUBJIDS: cell of subject IDs (characters). 
% EXPERIMENTTYPE: 'Discrim' or 'Detection'
% CONDITIONS: vector of condition numbers
% 
% ========== OUTPUT VARIABLES ==========
% PC: percent correct across conditions
%

% aspen yoo, aspen.yoo@nyu.edu
% 
% April 11, 2016
% updated some stuff. added PC. added annotations

% plot indvl data if only one..plot average if multiple subject IDs
nSubj = length(subjids);
if nSubj > 1; plotindvl = 0; else plotindvl = 1; end

% determining number of conditions from first subject
if strcmp(subjids{1}(1),'F'); % if fake data
    load(['fakedata_subj' subjids{1} '.mat'],'Xdet');
    data = Xdet;
    nCond = length(data);
else
    [data] = concatdata(subjids{1}, experimenttype);
    if strcmp(experimenttype,'Detection')
        condDivide = 2;
    else strcmp(experimenttype,'Discrim')
        condDivide = 1;
    end
    nCond = max(data(:,5))/condDivide;
end

if nargin < 3; conditions = 1:nCond; end
nCond = length(conditions);
colorMat = aspencolors(nCond,'blue');

halferrorbar = 1;
if (plotindvl) 
    
    if nargout == 1;
        pc = nan(1,nCond);
        for icond = 1:nCond;
            if condDivide == 2;
                corrResp = data(data(:,5) == icond*2,7);
                corrResp = [corrResp; data(data(:,5) == icond*2-1,7)];
                pc(icond) = mean(corrResp);
            else
                pc(icond) = mean(data(data(:,5) == icond,7));
            end
        end        
    end
    
    % load real data
    if iscell(data);        % if data are already split up by condition
        data = cellfun(@(x) x(x(:,1) ~= 0,:),data,'UniformOutput',false);
    else
        data(data(:,1) == 0, :) = []; % deleting trials with no change
    end
    [stimlevels, trialnums, nresps] = conditionSeparator(data,1);
    
    nLevels = length(stimlevels{1});
    hold on
    for icond = 1:nCond;
        cond = conditions(icond);

        for ilevel = 1:nLevels;
            currplot = plot(stimlevels{cond}(ilevel), nresps{cond}(ilevel)/trialnums{cond}(ilevel),'.');
            set(currplot,'MarkerSize', trialnums{cond}(ilevel),...
                'Color',       colorMat(cond,:));
            if ilevel ~= nLevels; set(get(get(currplot,'Annotation'),'LegendInformation'),...
                    'IconDisplayStyle','off'); end % Exclude line from legend
            alpha = nresps{icond}(ilevel) + 1;
            beta = trialnums{icond}(ilevel) - alpha + 2;
            varr = (alpha*beta)/((alpha+beta)^2*(alpha+beta+1));
            errorbar(stimlevels{cond}(ilevel),nresps{cond}(ilevel)/trialnums{cond}(ilevel),sqrt(varr),...
                'LineStyle','none','color',colorMat(cond,:));
            hold on
            plot([stimlevels{cond}(ilevel)-halferrorbar stimlevels{cond}(ilevel)+halferrorbar],...
                [nresps{cond}(ilevel)/trialnums{cond}(ilevel)-sqrt(varr) ...
                nresps{cond}(ilevel)/trialnums{cond}(ilevel)-sqrt(varr)],'color',colorMat(cond,:));
            plot([stimlevels{cond}(ilevel)-halferrorbar stimlevels{cond}(ilevel)+halferrorbar],...
                [nresps{cond}(ilevel)/trialnums{cond}(ilevel)+sqrt(varr) ...
                nresps{cond}(ilevel)/trialnums{cond}(ilevel)+sqrt(varr)],'color',colorMat(cond,:));
        end
        defaultplot;
        axis([-45 45 0 1])
        set(gca,'Ytick',[0 0.5 1],'Xtick',[-45 0 45])
        ylabel('p(respond same)');
        xlabel('orientation change (deg)');
% 	    legend(legendMat{conditions})
    end

else
    
    for isubj = 1:nSubj;
        subjid = subjids{isubj};
        
        % load data
        [data] = concatdata(subjid, experimenttype);
        data(data(:,1) == 0, :) = []; % deleting trials with no change
        [stimlevels, trialnums, nresps] = conditionSeparator(data,1);
        nLevels = length(stimlevels{1});
        
        pp = cellfun(@(x,y) (x./y), nresps, trialnums, 'UniformOutput', false);
        for icond = 1:nCond;
            cond = conditions(icond);
            pdiff{icond}(isubj,:) = pp{icond};
        end

    end
          
    mean_data = cellfun(@(x) mean(x),pdiff,'UniformOutput',false);
    sem_data = cellfun(@(x) std(x)/sqrt(nSubj),pdiff,'UniformOutput',false);
    
    for cond = 1:nCond;
        %         subplot(1,nCond,cond);
        databars = errorbar(stimlevels{cond},mean_data{cond},sem_data{cond});
        hold on;
        set(databars,'Color', colorMat(cond,:),...
            'LineStyle'     ,'none'    ,...
            'LineWidth'     , 1.5  );
        set(get(get(databars,'Annotation'),'LegendInformation'),...
            'IconDisplayStyle','off'); % Exclude line from legend
    end
    defaultplot
    axis([-40 40 0 1])
    xlabel('amount of change (deg)')
    
    switch experimenttype
        case 'Discrim'
            ylabel('proportion respond clockwise');
        case 'Detection'
            ylabel('proportion respond same');
    end

end