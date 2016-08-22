function [loglike,p_resp] = loglike_onecondition(data,bias,kappa,kappatilde,prior,lapserate)
% LOGLIKE_ONECONDITION calculates log likelihood of sameness of data from one condition 
% with Von Mises distributions
% 
% LOGLIKE = LOGLIKE_ONECONDITION(DATA,BIAS,KAPPA,KAPPATILDE,PRIOR,LAPSERATE)
% computes the log likelihood of the sameness/unity judgements to stimuli
% (second and first columns in DATA, respectively), given noise parameters
% KAPPA, internal noise likelihood parameters KAPPATILDE, prior
% parameters PRIOR, and lapse rate LAPSERATE. 
% 
% ============ INPUT VARIABLES ==============
% 
%  DATA is an nTrials x 2 matrix. First column has the value of change
%  (difference in orientation from first to second display). Second column
%  has the subject's response (1: same, 0: different) for the corresponding
%  trial. 
% 
%  BIAS and KAPPA are respecetively BIAS and KAPPA of objective sensory noise.
%  The noise is assumed to be Von Mises, centered on S (possibly with bias).
%
%  KAPPATILDE is the KAPPA of internal sensory likelihood.
%  The likelihood is assumed to be Von Mises, centered on the measurement.
%
%  PRIOR is a vector of parameters for the prior. 
%  For a Bayesian observer, PRIOR(1) and PRIOR(2) represent the mean and
%  concentration parameter of the Von Mises prior, and PRIOR(3) encodes 
%  PCOMMON (probability of sameness).
%  For a non-Bayesian observer, PRIOR(1) and PRIOR(2) are NaN's and
%  PRIOR(3) represents the non-Bayesian criterion.
%
%  LAPSERATE is the probability of a lapse. In each trial, the subject has
%  a probability in between 0 and 1 to ignore the stimuli and respond
%  same with probability 0.5. 


% stuff from luigi's code in Oct 16, 2015. 
%
%  LOGLIKE = AHYVSTM_DATALIKE_SAMENESS_VM(...,N) specifies the number of points 
%  of the grid for Riemannian integration. Larger N values make the 
%  computation slower but generally more precise (default N=500).
%
%  [LOGLIKE,P_RESP] = AHYVSTM_DATALIKE_SAMENESS_VM(...) also returns the vector 
%  P_RESP with the probabilities of each subject's response for each trial 
%  (useful for debugging). Note that to prevent numerical problems or
%  instabilities due to outliers, each response has a minimum probability 
%  of 10^-6.

Nx = 3601;              % Number of grid points
smoothflag = 1;         % Smooth likelihood landscape

% Convert to radians and stretch from [-pi/2,pi/2] to [-pi,pi], hence the 2
C = 2*pi/180;
prior_mu = C*prior(1);
prior_kappa = prior(2); % Concentration parameter of the VM prior over deltas
bias = C*bias;
stim = C*data(:,1);
resp = data(:,2);

% Grid for Riemannian integration
xrange = C*linspace(-90,90,Nx);
dx = xrange(2) - xrange(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute log P(x,s2|C=1), marginal likelihood of sameness.

Izero = @(x_) besseli(0,x_,1);  % Rescaled Bessel function of the 1st kind

% Decision variable
if all(isfinite(prior))     % Bayesian criterion

    pcommon = prior(3); % Probability of sameness P(C=1)    
    kappabar = sqrt(kappatilde^2 + prior_kappa^2 + 2*kappatilde*prior_kappa*cos(prior_mu - xrange));
    %d = log(pcommon/(1-pcommon)) - log(2*pi*Izero(kappatilde)) + kappatilde.*cos(xrange) ...
    %    + log(2*pi*Izero(kappatilde)*Izero(prior_kappa)) - log(Izero(kappabar));
    d = log(pcommon/(1-pcommon)) + kappatilde.*cos(xrange) ...
        + (prior_kappa-kappabar).*log(Izero(prior_kappa)./Izero(kappabar));  
    
    decision = d > 0; % all of the datapoints that are above 0.
    distance1 = -d(find(decision,1,'first')-1)/(d(find(decision,1,'first')) - d(find(decision,1,'first')-1));
    distance2 = d(find(decision,1,'last'))/(d(find(decision,1,'last')+1) - d(find(decision,1,'last')));
    criteria(1) = xrange(find(decision,1,'first')-1) + dx*distance1;
    criteria(2) = xrange(find(decision,1,'last')) + dx*distance2;

    
    if smoothflag
        delta = 0.01;
        d = 1./(1 + exp(-d/delta));
    else
        d = d >= 0;
    end
else                        % Non-Bayesian criterion on x
    
    kcommon = C*prior(3);
    if smoothflag
        delta = dx/3;
        d = (xrange - kcommon).*(xrange >= 0) + (-xrange - kcommon).*(xrange < 0);
        d = 1 - 1./(1 + exp(-d/delta));
    else
        d = xrange >= -kcommon & xrange <= kcommon;
    end
    
end

index = d > eps;

% Probability of response \int [C_hat(x) == 1] p(x|s) dx
if smoothflag
    pc_one = qtrapz(bsxfun(@times,exp(kappa*(cos(bsxfun(@minus, xrange(index) - bias, stim))-1)),d(index)),2);
else
    pc_one = qtrapz(exp(kappa*(cos(bsxfun(@minus, xrange(d) - bias, stim))-1)),2);
end
pc_one = dx*pc_one./(2*pi*Izero(kappa));

p_resp = resp.*pc_one + (1-resp).*(1-pc_one);

% Compute response lapse (if any). 
if lapserate > 0
    % Prior-dependent lapse (when lapsing, respond according to P(C=1))
    %p_resp = lapserate*(pcommon.*R + (1-pcommon).*(1 - R)) + (1-lapserate)*p_resp;

    % Standard fifty-fifty lapse
    p_resp = lapserate*0.5 + (1-lapserate)*p_resp;
end

% Finalize log likelihood.
MIN_P = 1e-10; % Cutoff to the penalty for a single outlier
p_resp = MIN_P + (1-MIN_P)*p_resp;
loglike = sum(log(p_resp));

end