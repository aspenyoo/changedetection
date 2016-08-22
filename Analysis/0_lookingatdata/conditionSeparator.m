function varargout = conditionSeparator(designMat,isgaussprior,deletezeros)
% separates data based on conditions and gives back only delta and response
if nargin < 2; isgaussprior = 1; end
if nargin < 3; deletezeros = 1; end

if iscell(designMat);
    data = designMat;
    nCond = length(data);
    det_scalefactor = 1; 
else
    nCond =  max(designMat(:,5));
    if any(designMat(:,6) == -1);
        det_scalefactor = 1;
    else
        det_scalefactor = 2;
    end
    setsizeNum = unique(designMat(:,2));
    reliabilityNum = unique(designMat(:,3));
    delaytimeNum = unique(designMat(:,4));
    
    nCondPerIV = [length(setsizeNum) length(reliabilityNum) length(delaytimeNum)];
    
    if prod(nCondPerIV)*det_scalefactor ~= nCond;
        error('number of conditions do not match')
    end
    
    [setsizeVec,relVec,deltimeVec] = meshgrid(setsizeNum, reliabilityNum, delaytimeNum);
    
    data = cell(1,nCond/det_scalefactor);
    for i = 1:nCond;
        
        ii = ceil(i/det_scalefactor);
        setsize = setsizeVec(ii);
        rel = relVec(ii);
        deltime = deltimeVec(ii);
        
        dMat = designMat;
        dMat = dMat(setsize == dMat(:,2),:);
        dMat = dMat(rel == dMat(:,3),:);
        dMat = dMat(deltime == dMat(:,4),:);
        %     if det_scalefactor == 2;
        %         if mod(i,2);
        %             dMat(dMat(:,1) ~= 0,:) = [];
        %         else
        %             dMat(dMat(:,1) == 0,:) = [];
        %         end
        %     end
        if isempty(data{ii})
            data{ii} = dMat(:,[1 6]);
        end
        data{ii}(data{ii}(:,2) == -1,2) = 0;
        
    end
end

if nargout == 1;
    varargout{1} = data;
else
    
    if (isgaussprior)
        gaussprior = 20; % 0: uniform prior. otherwise, gaussian prior
    else
        nTrialsPerLevel = 30;
    end
    
    trialNumsCell = cell(1,nCond/det_scalefactor);
    stimLevelsCell = cell(1,nCond/det_scalefactor);
    nRespCell = cell(1,nCond/det_scalefactor);
    for icond = 1:(nCond/det_scalefactor)
        
        data{icond} = sortrows(data{icond});
        deltas = data{icond}(:,1);
        responses = data{icond}(:,2);
        
        % ========== BINNING/SETTING UP DATA ==========
        % binning if not set levels
        if size(unique(deltas),1) > 10;
            if (isgaussprior); % nLevels determined by average.
                nLevels = 10; % 1 more than actual bins you want
                Levels = gaussprior*norminv(linspace(0,1,nLevels));
                Levels = Levels(2:end-1);
            else % uniform prior
                nLevels = ceil(size(deltas,1)/nTrialsPerLevel);
            end
            
            [trialNums, stimLevels] = aspenhistc(deltas,Levels);
            deltaResp1 = deltas(responses==1);
            [nResp] = aspenhistc(deltaResp1,Levels);
            %             assert(sum(stimLevels == stimLevels2) == length(stimLevels));
            %             sizeBin_half = (stimLevels(3) - stimLevels(2))/2;
            %             nResp = nan(1,nLevels);
            %             nResp(1) = sum(responses(deltas < (stimLevels(1) + sizeBin_half)));
            %             nResp(end) = sum(responses(deltas > (stimLevels(end) - sizeBin_half)));
            %             for ilevel = 2:nLevels-1;
            %                 nResp(ilevel) = sum(responses((deltas < (stimLevels(ilevel)+sizeBin_half)) & ...
            %                     (deltas > (stimLevels(ilevel)-sizeBin_half))));
            %             end
            %             counter = 1;
            %             stimLevels = nan(1,nLevels); nResp = nan(1,nLevels);
            %             for i = 1:nLevels-1;
            %                 stimLevels(i) = mean(deltas(counter:counter+nTrialsPerLevel));
            %                 nResp(i) = sum(responses(counter:counter+nTrialsPerLevel));
            %                 counter = counter + nTrialsPerLevel;
            %             end
            %             stimLevels(end) = mean(deltas(counter:end));
            %             nResp(end) = sum(responses(counter:end));
            %             trialNums = nTrialsPerLevel*ones(1,nLevels);
            %             trialNums(end) = nTrialsPerLevel + mod(size(deltas,1),nTrialsPerLevel*(nLevels));
        else
            stimLevels = unique(deltas);
            nLevels = length(stimLevels);
            trialNums = nan(1,length(stimLevels)); nResp = trialNums;
            for ilevel = 1:nLevels;
                trialNums(ilevel) = sum(deltas == stimLevels(ilevel));
                nResp(ilevel) = sum(responses(deltas == stimLevels(ilevel)));
            end
        end
        
        trialNumsCell{icond} = trialNums;
        stimLevelsCell{icond} = stimLevels;
        nRespCell{icond} = nResp;
    end
    
    varargout{1} = stimLevelsCell;
    varargout{2} = trialNumsCell;
    varargout{3} = nRespCell;
end