function [ll,p_resp] = loglike(data, model, theta)
% LOGLIKE(DATA, MODEL, THETA) calculated
% 
% THETA: some theta values are given in log format, so they are
% exponentiated here. 

nCond = length(data); % number of conditions

% kappa_prior = 8.742;
kappa_prior = 2.6982; % REAL KAPPA!! (goes from -pi to pi instead of -pi/2 to pi/2) 4/28/16

bias = 0;
kappaVec = exp(theta(1:nCond));
lapserate = theta(end);
switch model
    case 1 % Optimal model, only free noise and lapse
        kappatildeVec = kappaVec;
        prior = [0 kappa_prior 0.5];% true prior! mean and kappa of change dist. pcommon
    case 2 % Optimal model with free prior on (C=1)
        kappatildeVec = kappaVec;
        prior = [0 kappa_prior theta(end-1)];% true prior! mean and kappa of change dist. pcommon
    case 3 % Optimal model with free prior on C and free prior on S
        kappatildeVec = kappaVec;
        prior = [0 exp(theta(end-1)) theta(end-2)];% true prior! mean and kappa of change dist. pcommon
    case 4 % Optimal model with free prior on S
        kappatildeVec = kappaVec;
        prior = [0 theta(end-1) 0.5];% true prior! mean and kappa of change dist. pcommon
    case 5 % Fixed criterion model, free noise and sigmatilde and lapse
        kappatildeVec = exp(theta(end-1))*ones(1,nCond);
        prior = [0 kappa_prior 0.5];% true prior! mean and kappa of change dist. pcommon
    case 6 % Super-free model, free noise and full free sigmatilde and lapse
        kappatildeVec = exp(theta(nCond+1:2*nCond));
        prior = [0 kappa_prior 0.5];% true prior! mean and kappa of change dist. pcommon
    case 7 % linear heuristic model
        kappatildeVec = kappaVec; % not actually used in LL calculation.
        kcommonVec = theta(nCond+1:nCond+2); % decision boundary in degrees (low and high kcommon)
        prior = [Inf Inf nan]; % Inf indicates to use non-Bayesian criteria
end

ll = nan(1,nCond); p_resp = cell(1,nCond);
for icond = 1:nCond;
    kappa = kappaVec(icond);
    kappatilde = kappatildeVec(icond);
    currcondData = data{icond};
    if model == 7; % if non-Bayesian criteria
%         prior(3) = (kcommonVec(2)-kcommonVec(1))*(kappa-kappaVec(1))+kcommonVec(1) %kcommon = mx + b OLD ONE 09/15/2016
%         
%         blah = (kcommonVec(2)-kcommonVec(1))*(kappaVec(5)-kappaVec(1))+kcommonVec(1);
        prior(3) = (kcommonVec(2)-kcommonVec(1))/(kappaVec(5)-kappaVec(1))*(kappa-kappaVec(1))+kcommonVec(1); %kcommon = mx + b
    end
%     [loglike(icond),p_resp{icond}] = loglike_onecondition(currcondData,bias,kappa,kappatilde,prior,lapserate);
    [ll(icond),p_resp{icond}] = AhyVSTM_datalike_sameness_VM(currcondData(:,1),currcondData(:,2),bias,kappa,kappatilde,prior,lapserate);
end

ll = sum(ll);

