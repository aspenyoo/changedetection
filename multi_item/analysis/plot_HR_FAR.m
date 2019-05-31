function [HRallVec,HRlowVec,HRhighVec,FARVec,mod_HRallVec,mod_HRlowVec,mod_HRhighVec,mod_FARVec] = plot_HR_FAR(data,prediction,isplot)

if nargin < 2; prediction = []; end
if nargin < 3; isplot = 1; end


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

% get colormap info


[HRallVec,HRlowVec,HRhighVec,FARVec] = deal(nan(1,length(n_high_vec)));
[mod_HRallVec,mod_HRlowVec,mod_HRhighVec,mod_FARVec] = deal(nan(1,length(n_high_vec)));
for ihigh = (n_high_vec+1)
    % get indices of current reliability number
    idx_start = idx_high(ihigh)+1;       % which row starts this n_high
    idx_stop = idx_high(ihigh+1);        % end of this thing
    
    % get subset of relevant data
    subdata = [data.Delta(idx_start:idx_stop) data.resp(idx_start:idx_stop)];    
    delta_low = sum(Delta(idx_start:idx_stop,1:(5-ihigh)),2);
    if ~isempty(prediction); subdata = [subdata prediction(idx_start:idx_stop)]; end
   
    % get FAR
    idx_nochange = subdata(:,1) < eps;
    FARVec(ihigh) = mean(subdata(idx_nochange,2));
    if ~isempty(prediction); mod_FARVec(ihigh) = mean(subdata(idx_nochange,3)); end
    
    % get HR
    idx_change = subdata(:,1) >= eps;
    HRallVec(ihigh) = mean(subdata(idx_change,2));
    idxx = (delta_low >= eps); % change in low rel ellipse
    HRhighVec(ihigh) = mean(subdata((~idxx & idx_change),2));
    HRlowVec(ihigh) = mean(subdata(idxx,2));
    if ~isempty(prediction); 
        mod_HRallVec(ihigh) = mean(subdata(idx_change,3));
        mod_HRhighVec(ihigh) = mean(subdata((~idxx & idx_change),3));
        mod_HRlowVec(ihigh) = mean(subdata(idxx,3));
    end
    
end

if (isplot)
    hold on;
    plot(n_high_vec,FARVec,'o-')
    plot(n_high_vec,HRallVec,'o-')
    plot(n_high_vec,HRhighVec,'o-')
    plot(n_high_vec,HRlowVec,'o-')
    legend('FAR','HR all','HR high', 'HR low')
    
    if ~isempty(prediction); 
        plot(n_high_vec,mod_FARVec,'--')
        plot(n_high_vec,mod_HRallVec,'--')
        plot(n_high_vec,mod_HRhighVec,'--')
        plot(n_high_vec,mod_HRlowVec,'--')
    end
    
    xlabel('amount of orientation change')
    ylabel('proportion report change')
    
    defaultplot
    axis([-0.5 4.5 0 1])
    set(gca,'XTick',0:4,'XTickLabel',0:4,'YTick',0:0.2:1,'YTickLabel',0:0.2:1)
end

end