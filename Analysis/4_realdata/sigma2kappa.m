function kappa = sigma2kappa(sigma)
%SIGMA2KAPPA Convert from Gaussian SD to Von Mises concentration parameter.
%
%   KAPPA = SIGMA2KAPPA(SIGMA) convert Gaussian standard deviation SIGMA 
%   (expressed in degrees) into Von Mises's concentration parameter KAPPA.
%   SIGMA is expressed in degrees and it is assumed a range from -180 to
%   180 degrees (multiply SIGMA by two if the allowed range is from -90 to
%   90 degrees). 
%
%   The conversion keeps the Fisher information (FI) constant, knowing that
%   FI(sigma) = 1/sigma^2    and   FI(kappa) = kappa*I_1(kappa)/I_0(kappa)
%   where I_j is the modified bessel function of the first kind.
%
%   SIGMA2KAPPA uses a lookup table that is computed once and stored in 
%   memory (via a persistent variable) to prevent further computations.

persistent lookuptable;

% Compute lookup table with mapping from J (Fisher information) 
% to kappa (concentration parameter of Von Mises)
if isempty(lookuptable)
    % Lookup table for large values of J (almost linear)
    kapparangeLarge = linspace(1,1e4,1e5);
    JrangeLarge = kapparangeLarge.*besseli(1,kapparangeLarge,1)./besseli(0,kapparangeLarge,1);
    lookuptable.JrangeLarge = linspace(1,JrangeLarge(end),1e6);
    lookuptable.kappainvLarge = interp1(JrangeLarge,kapparangeLarge,lookuptable.JrangeLarge,'pchip');

    % Lookup table for small values of J
    kapparangeSmall = linspace(0,2,1e4);
    JrangeSmall = kapparangeSmall.*besseli(1,kapparangeSmall,1)./besseli(0,kapparangeSmall,1);
    lookuptable.JrangeSmall = linspace(0,1,1e4);
    lookuptable.kappainvSmall = interp1(JrangeSmall,kapparangeSmall,lookuptable.JrangeSmall,'pchip');
end

% Equivalent Fisher information of a Gaussian (in radians)
J = 1/(sigma/180*pi)^2;

% Compute kappa depending on the case

% Very large J: kappa(J) is approximately linear (plus an offset)
if J >= lookuptable.JrangeLarge(end)
    kappa = lookuptable.kappainvLarge(end) + (J - lookuptable.JrangeLarge(end));    
    
% J greater than 1 (high precision)   
elseif J >= 1
    dJ = lookuptable.JrangeLarge(2)-lookuptable.JrangeLarge(1);
    idx = 1 + (J - lookuptable.JrangeLarge(1))/dJ; 
    idx1 = floor(idx);  idx2 = ceil(idx);
    w = idx - idx1;
    kappa = (1-w)*lookuptable.kappainvLarge(idx1) + w*lookuptable.kappainvLarge(idx2);
    
% J between 0 and 1 (low precision)
else
    dJ = lookuptable.JrangeSmall(2)-lookuptable.JrangeSmall(1);
    idx = 1 + (J - lookuptable.JrangeSmall(1))/dJ; 
    idx1 = floor(idx);  idx2 = ceil(idx);
    w = idx - idx1;
    kappa = (1-w)*lookuptable.kappainvSmall(idx1) + w*lookuptable.kappainvSmall(idx2);        
end

end