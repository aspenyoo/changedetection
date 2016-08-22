function [loglike,p_resp] = AhyVSTM_datalike_sameness_VM(S,R,bias,kappa,kappatilde,prior,lapserate)
%AHYVSTM_DATALIKE_SAMENESS_VM Log likelihood of sameness with Von Mises distributions (AHY VSTM experiment).
%
%  LOGLIKE = AHYVSTM_DATALIKE_SAMENESS_VM(S,R,KAPPA,KAPPATILDE,PRIOR) computes the 
%  log likelihood of sameness/unity judgments R to stimuli S, given noise 
%  parameters KAPPA, internal noise likelihood parameters KAPPATILDE and 
%  prior parameters PRIOR.
%
%  S is a M x 1 vector (M number of trials). Each element is the value
%  of the difference in orientation between test and reference stimulus
%  (that is, s1 - s2). That is, the reference frame is always centered on
%  the reference stimulus.
%
%  R is a M x 1 vector of categorical subject's responses. 1 means
%  perceived sameness, 0 perceived difference.
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
%  LOGLIKE = AHYVSTM_DATALIKE_SAMENESS_VM(S,R,KAPPA,KAPPATILDE,PRIOR,LAPSERATE) 
%  includes a probability of lapse. In each trial, the subject has a 
%  probability 0<=LAPSERATE<=1 to ignore the stimuli and simply respond 
%  either way with 50% probability.
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

% Luigi Acerbi <luigi.acerbi@gmail.com>
% First version: Oct/16/2015.
% This is a template -- when you make changes, please rename this file.

Nx = 3601;              % Number of grid points
smoothflag = 1;         % Smooth likelihood landscape

% Convert to radians and stretch from [-pi/2,pi/2] to [-pi,pi], hence the 2
C = 2*pi/180;
prior_mu = C*prior(1);
prior_kappa = prior(2); % Concentration parameter of the VM prior over deltas
bias = C*bias;
S = C*S;

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
%     d = log(pcommon/(1-pcommon)) - log(2*pi*Izero(kappatilde)) + kappatilde.*cos(xrange) ...
%        + log(2*pi*Izero(kappatilde)*Izero(prior_kappa)) - log(Izero(kappabar)); % luigi old comment
    d = log(pcommon/(1-pcommon)) + kappatilde.*cos(xrange) ...
       +(prior_kappa-kappabar).*log(Izero(prior_kappa)./Izero(kappabar)); % luigi current
%     d = log(pcommon/(1-pcommon)) + kappatilde.*cos(xrange) + log(Izero(prior_kappa))...
%         -log(Izero(kappabar)); % aspen current (apr 18, 2016) THIS SI
%         WRONG! BC Izero
    
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
    pc_one = qtrapz(bsxfun(@times,exp(kappa*(cos(bsxfun(@minus, xrange(index) - bias, S))-1)),d(index)),2);
else
    pc_one = qtrapz(exp(kappa*(cos(bsxfun(@minus, xrange(d) - bias, S))-1)),2);
end
pc_one = dx*pc_one./(2*pi*Izero(kappa));

p_resp = R.*pc_one + (1-R).*(1-pc_one);

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