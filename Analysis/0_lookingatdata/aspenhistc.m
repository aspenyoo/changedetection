function [no, mid] = aspenhistc(X,edges)
% ASPENHISTC modified HISTC function
% 
%   [N,MID] = ASPENHISTC(X,EDGES), for vector X, counts the number of values in X
%   that fall between the elements in the EDGES vector (which must contain
%   monotonically non-decreasing values).  N is a LENGTH(EDGES)+1 vector
%   containing these counts.  
%
%   N(k) will count the value X(i) if EDGES(k) <= X(i) < EDGES(k+1).  The
%   first bin will count any values of X less than EDGES(1), and the last
%   bin will count any values greater than or equal to EDGES(end).
%   
%   MID(k) will output the mean value of the samples within each bin. Any
%   infs or nans will be ignored. 

X = sort(X(:));
edges = [-Inf edges(:)' Inf];

nbins = length(edges) - 1;
no = nan(1,nbins);
mid = nan(1,nbins);
for ibin = 1:nbins;
    currvals = X((X >= edges(ibin)) & (X < edges(ibin+1)));
    no(ibin) = max(size(currvals));
    mid(ibin) = mean(currvals);
end