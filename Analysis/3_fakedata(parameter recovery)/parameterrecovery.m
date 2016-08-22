function [trueTheta, estTheta, nLL_est] = parameterrecovery(model,nSubj)

nCond = 5;
nStartVals = 50;

% get reasonable parameter values 
switch model
    case 1 % Optimal model, only free noise and lapse
        trueTheta = [cumsum(rand(nSubj, nCond),2), rand*0.1];
    case {2, 4} % Optimal model with free prior on (C=1)
        trueTheta = [cumsum(rand(nSubj, nCond),2), 0.25+rand*0.5 rand*0.1];
    case 3 % Optimal model with free prior on C and free prior on S
        trueTheta = [cumsum(rand(nSubj, nCond),2), 0.25+rand*0.5 log(15+rand*10) rand*0.1];
    case 5 % M2D Fixed criterion model, free noise and sigmatilde and lapse
        trueTheta = [cumsum(rand(nSubj, nCond),2), 5*rand, rand*0.1];
    case 6 % M2A Super-free model, free noise and full free sigmatilde and lapse
        trueTheta = [cumsum(rand(nSubj, nCond),2), cumsum(rand(nSubj, nCond),2), rand*0.1];
end

nParams = size(trueTheta,2);
estTheta = nan(nSubj,nParams);
nLL_est = nan(1,nSubj);
for isubj = 1:nSubj;
    
    truetheta = trueTheta(isubj,:);
    
    % simulate data
    [data] = simulateresp(model, truetheta);
    
    % fit data
    [estTheta(isubj,:), nLL_est(isubj)] = fitparam(data, model, nStartVals);
    
end