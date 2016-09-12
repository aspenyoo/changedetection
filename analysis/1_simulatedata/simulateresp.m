function [Xdet] = simulateresp(model, theta, nTrials)
if nargin < 3; nTrials = 600; end
% if nargin < 3; gaussprior = 20; end

vmprior = 8.742; % kappa. VM prior

% prior = [0 8.742 0.5];

nCond = 5;
switch model
    case 1 % Optimal model, only free noise and lapse
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = kappaVec;
        prior = [0 8.742 0.5];% true prior! mean and kappa of change dist. pcommon
        lapserate = theta(end);
    case {2, 4} % Optimal model with free prior on (C=1)
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = kappaVec;
        prior = [0 8.742 theta(end-1)];% true prior! mean and kappa of change dist. pcommon
        lapserate = theta(end);
    case 3 % Optimal model with free prior on C and free prior on S
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = kappaVec;
        prior = [0 theta(end-1) theta(end-2)];% true prior! mean and kappa of change dist. pcommon
        lapserate = theta(end);
    case 5 % M2D Fixed criterion model, free noise and sigmatilde and lapse
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = theta(end-1)*ones(1,nCond);
        prior = [0 8.742 0.5];% true prior! mean and kappa of change dist. pcommon
        lapserate = theta(end);
    case 6 % M2A Super-free model, free noise and full free sigmatilde and lapse
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = theta(nCond+1:2*nCond);
        prior = [0 8.742 0.5];% true prior! mean and kappa of change dist. pcommon
        lapserate = theta(end);
    case 7 % linear heuristic model
        bias = 0;
        kappaVec = theta(1:nCond);
        kappatildeVec = nan(1,nCond); % not used in actual calculation
        kcommonVec = theta(nCond+1:nCond+2);
        prior = [Inf Inf nan];
        lapserate = theta(end);
end

resp = ones(nTrials,1);

p_resp = cell(1,nCond);
Xdet = cell(1,nCond);
for icond = 1:nCond;
    stim = [zeros(nTrials/2,1); circ_vmrnd(0,vmprior, [nTrials/2,1])*180/pi];
    kappa = kappaVec(icond);
    kappatilde = kappatildeVec(icond);
    if model == 7; % if non-Bayesian criteria
        prior(3) = (kcommonVec(2)-kcommonVec(1))*(kappa-kappaVec(1))+kcommonVec(1); %kcommon = mx + b
    end
    [~,p_resp{icond}] = AhyVSTM_datalike_sameness_VM(stim,resp,bias,kappa,kappatilde,prior,lapserate);
    Xdet{icond} = [stim binornd(ones(nTrials,1),p_resp{icond})];
end

% [~,~,P_resp] = AhyVSTM_datalike(Xdsc,Xdet,mu,sigma,sigmatilde,prior,lapserate);