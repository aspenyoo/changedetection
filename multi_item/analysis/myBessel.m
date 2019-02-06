% faster lookup version of bessel function

function [output] = myBessel(X,spacing,lookupY)

% Do for each row (Kappas must be aligned in rows)

% Reshape X into one column with columns concatenated for qinterp1
tempX = reshape(X,[],1);

% Find what is greater than 5
%TXG5 = tempX.*(tempX>5);

%TXL5 = tempX<=5;
% get spacing

% Do the interpolation on everything less than 5
% L5 = (qinterp1(lookup(:,1),lookup(:,2),tempX));
L5 = qinterp1(spacing,lookupY,tempX);

% Use large bessel approx for everything greater than 5
%G5 = ((exp(TXG5)./(sqrt(2*pi*TXG5))).*(1+1./(8*TXG5))).*(tempX>5);
%G5((isnan(G5)))=0;

% Combine the values
% output = L5; %+ G5;

output = reshape(L5,size(X));
