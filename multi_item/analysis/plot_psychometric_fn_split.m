function [x_mean, pc_data, pc_pred] = plot_psychometric_fn_split(data,nBins,prediction,quantilebinedges)
%PLOT_PSYCHOMETRIC_FN plots proportion report change as a function of
%number of high reliability items and amount of change
% 
% ================ INPUT VARIABLES ==============
% DATA: struct
%   follows format from 'data/fitting_data' files
% NBINS: scalar
%   number of bins desired
% PREDICTION: nTrials x 1 vector ([] default)
%   probability of responding change on every trial (matches data)
% QUANTILEBINEDGES: binary (1, default)
%   whether you want the bins of "change trials" to determined by
%   quantiling trials (1), or be the same across conditions (0)
% 
% ================= OUTPUT VARIABLES ================
% X_MEAN: nRels x nBins matrix
%   average delta for current bin
% PC_DATA: nRels x nBins matrix
%   average proportion respond change for current bin
% PC_PRED: nRels x nBins matrix
%   average predicted proportion of responding change

if nargin < 2; nBins = 6; end
if nargin < 3; prediction = []; end
% if nargin < 4; quantilebinedges = 1; end

nItems = 4;
n_high_vec = 0:nItems;

% get indices of sections of data with nItems
% note: data must be sorted already by number of high reliability items
rels = unique(data.rel);
idx_high = [0 nan(1,nItems+1)];
for n_high = n_high_vec;
    idx_high(n_high+2) = find(sum(data.rel ==rels(2),2)==n_high,1,'last');
end

% change delta to be change, and up to pi/2
Delta = abs(data.Delta);
data.Delta = 0.5*abs(circ_dist(sum(data.Delta,2),0));
binedges = linspace(eps,pi/2,nBins);

% % get colormap info
% h = figure(99);
% cmap = colormap('parula'); % get a rough colormap
% close(h)
% idxs = round(linspace(1,size(cmap,1),length(n_high_vec)));
% colorMat = cmap(idxs,:);

[x_mean, pc_data, pc_pred] = deal(zeros(length(n_high_vec),nBins-1,2));
for ihigh = 1:length(n_high_vec)
    % get indices of current reliability number
    idx_start = idx_high(ihigh)+1;       % which row starts this n_high
    idx_stop = idx_high(ihigh+1);        % end of this thing
    
    % get subset of relevant data
    Subdata = [data.Delta(idx_start:idx_stop) data.resp(idx_start:idx_stop)];    
    delta_low = sum(Delta(idx_start:idx_stop,1:(5-ihigh)),2);
    delta_high = sum(Delta(idx_start:idx_stop,(6-ihigh):4),2);
    if ~isempty(prediction); Subdata = [Subdata prediction(idx_start:idx_stop)]; end
    
    for irel = 1:2;
        
        % get indices
        if (irel == 1) % low-rel change
            idxx = (delta_low >= eps); % change in low rel ellipse
        else          % high-rel change
            idxx = (delta_high >= eps); % change in low rel ellipse
        end
        
        % sort rows by delta
        subdata = Subdata(idxx,:);
        subdata = sortrows(subdata,1);
        
%         % get mean for all no change trial
%         idx_nochange = find(subdata(:,1)<eps,1,'last');
%         pc_data(ihigh,1) = mean(subdata(1:idx_nochange,2));
%         if ~isempty(prediction); pc_pred(ihigh,1) = mean(subdata(1:idx_nochange,3));end
        
        % bin the rest of the change trials
        if (quantilebinedges) % get binedges based on number of trials (quantile binning)
            binedges = round(linspace(0,size(subdata,1),nBins+1));
        end
        for ibin = 1:nBins
            if (quantilebinedges)
                idxs_bin = (binedges(ibin)+1):(binedges(ibin+1));
            else
                idxs_bin = find((subdata(:,1)>=binedges(ibin)) & (subdata(:,1)<binedges(ibin+1)));
            end
            x_mean(ihigh,ibin,irel) = mean(subdata(idxs_bin,1));
            pc_data(ihigh,ibin,irel) = mean(subdata(idxs_bin,2));
            if ~isempty(prediction); pc_pred(ihigh,ibin,irel) =  mean(subdata(idxs_bin,3)); end
        end
    end
    
%     % plot
%     hold on;
%     plot(x_mean(ihigh,:),pc_data(ihigh,:),'o-','Color',colorMat(ihigh,:))
%     if ~isempty(prediction); plot(x_mean(ihigh,:),pc_pred(ihigh,:),'Color',colorMat(ihigh,:)); end
end

% xlabel('amount of orientation change')
% ylabel('proportion respond change')
% defaultplot

end