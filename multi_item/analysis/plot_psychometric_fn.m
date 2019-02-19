function plot_psychometric_fn(data,nBins,prediction)

if nargin < 2; nBins = 6; end
if nargin < 3; prediction = []; end

nItems = 4;
n_high_vec = 0:nItems;

% get indices of sections of data with nItems
% note: data must be sorted already by number of high reliability items
idx_high = [0 nan(1,nItems+1)];
for n_high = n_high_vec;
    idx_high(n_high+2) = find(sum(data.rel == 0.9,2)==n_high,1,'last');
end

% change delta to be absolute change, and up to pi/2
data.Delta = abs(mod(data.Delta,pi/2));

% get colormap info
h = figure(99);
cmap = colormap('parula'); % get a rough colormap
close(h)
idxs = round(linspace(1,size(cmap,1),length(n_high_vec)));
colorMat = cmap(idxs,:);


for ihigh = 1:length(n_high_vec)
    % get indices of current reliability number
    idx_start = idx_high(ihigh)+1;       % which row starts this n_high
    idx_stop = idx_high(ihigh+1);        % end of this thing
    
    % get subset of relevant data
    subdata = [sum(data.Delta(idx_start:idx_stop,:),2) data.resp(idx_start:idx_stop)];
    if ~isempty(prediction); subdata = [subdata prediction(idx_start:idx_stop)]; end
    
    % sort rows by delta
    subdata = sortrows(subdata,1);
    [x_mean, pc_data, pc_pred] = deal(nan(1,nBins));
    
    % get mean for all no change trial
    idx_nochange = find(subdata(:,1)==0,1,'last');
    x_mean(1) = 0;
    pc_data(1) = mean(subdata(1:idx_nochange,2));
    if ~isempty(prediction); pc_pred(1) = mean(subdata(1:idx_nochange,3));end
    
    % bin the rest of the change trials
    binedges = round(linspace(idx_nochange+1,size(subdata,1),nBins));
    for ibin = 1:(nBins-1)
        idxs_bin = (binedges(ibin)+1):(binedges(ibin+1));
        x_mean(ibin+1) = mean(subdata(idxs_bin,1));
        pc_data(ibin+1) = mean(subdata(idxs_bin,2));
        if ~isempty(prediction); pc_pred(ibin+1) =  mean(subdata(idxs_bin,3)); end
    end
    
    % plot
    hold on;
    plot(x_mean,pc_data,'o-','Color',colorMat(ihigh,:))
    if nargin > 1; plot(x_mean,pc_pred,'Color',colorMat(ihigh,:)); end
end


end