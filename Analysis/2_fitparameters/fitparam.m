function [bestFitParam, nLL_est, exitflag, output] = fitparam(data, model, nStartVals)
if nargin < 3; nStartVals = 1; end

nCond = length(data);
switch model
    case 1 % Optimal model, only free noise and lapse
        lb = [1e-2*ones(1,nCond) 0];
        ub = [200*ones(1,nCond) 1];
        plb = [1e-2*ones(1,nCond) 0];
        pub = [35*ones(1,nCond) 0.1];
        logflag = logical([ones(1,nCond) 0]);
    case 2 % Optimal model + free prior on (C=1)
        lb = [1e-2*ones(1,nCond) 0 0];
        ub = [200*ones(1,nCond) 1 1];
        plb = [1e-2*ones(1,nCond) 0 0];
        pub = [35*ones(1,nCond) 1 0.1];
        logflag = logical([ones(1,nCond) 0 0]);
    case 3 % Optimal model + free prior on C + free prior on S
        lb = [1e-2*ones(1,nCond) 0 1e-2 0];
        ub = [200*ones(1,nCond) 1 200 1];
        plb = [1e-2*ones(1,nCond) 0 1e-2 0];
        pub = [35*ones(1,nCond) 1 35 0.1];
        logflag = logical([ones(1,nCond) 0 1 0]);
    case 4 % Optimal model + free prior on S
        lb = [1e-2*ones(1,nCond+1) 0];
        ub = [200*ones(1,nCond+1) 1];
        plb = [1e-2*ones(1,nCond+1)  0];
        pub = [35*ones(1,nCond+1) 0.1];
        logflag = logical([ones(1,nCond+1) 0]);
    case 5 % Fixed criterion model: free noise, sigmatilde, lapse
        lb = [1e-2*ones(1,nCond+1) 0];
        ub = [200*ones(1,nCond+1) 1];
        plb = [1e-2*ones(1,nCond+1) 0];
        pub = [35*ones(1,nCond+1) 0.1];
        logflag = logical([ones(1,nCond+1) 0]);
    case 6 % Super-free model: free noise and full free sigmatilde and lapse
        lb = [1e-2*ones(1,2*nCond) 0];
        ub = [200*ones(1,2*nCond) 1];
        plb = [1e-2*ones(1,2*nCond) 0];
        pub = [35*ones(1,2*nCond) 0.1];
        logflag = logical([ones(1,2*nCond) 0]);
    case 7 % linear heuristic model: free noise, kcommon_low, kcommon_high, lapse
        lb = [1e-2*ones(1,nCond) zeros(1,2) 0];
        ub = [200*ones(1,nCond) 90*ones(1,2) 1];
        plb = [1e-2*ones(1,nCond) zeros(1,2) 0];
        pub = [35*ones(1,nCond) 90*ones(1,2) 0.1];
        logflag = logical([ones(1,nCond) zeros(1,3)]);
end
lb(logflag) = log(lb(logflag));
ub(logflag) = log(ub(logflag));
plb(logflag) = log(plb(logflag));
pub(logflag) = log(pub(logflag));

% optimization options
% options = bps('defaults');              % Default options
% options.UncertaintyHandling = 0;        % Activate noise handling
% options.NoiseSize = 1;                  % Estimated noise magnitude
options = optimset('Display','iter');

optfunc = @(x)-loglike(data, model, x);

nParams = length(lb);
bfp = nan(nStartVals,nParams); nll = nan(1,nStartVals); exitflag = nan(1,nStartVals); outputt = cell(1,nStartVals);
for istartval = 1:nStartVals;
    
    startTheta = plb+rand(1,nParams).*(pub-plb); % startTheta based on plausible ub and lb
    
%     [bfp(istartval,:) ,nll(istartval), exitflag(istartval), outputt{istartval}] = bps(optfunc,startTheta,lb,ub,plb,pub,options);
    [bfp(istartval,:) ,nll(istartval), exitflag(istartval), outputt{istartval}] = fmincon(optfunc,startTheta,[],[],[],[],lb,ub,[],options);
end
% bfp
% exitflag
% outputt
% 
% while exitflag(nll == min(nll)) == 0;
%     bfp(nll == min(nll)) = [];
%     exitflag(nll == min(nll)) = [];
%     outputt(nll == min(nll)) = [];
%     nll(nll == min(nll)) = [];
% end
% outputt

bestFitParam = bfp(nll == min(nll),:);
exitflag = exitflag(nll == min(nll));
output = outputt{nll == min(nll)};
nLL_est = nll(nll == min(nll));
bestFitParam(logflag) = exp(bestFitParam(logflag)); % exponentiating appropriate parameters back