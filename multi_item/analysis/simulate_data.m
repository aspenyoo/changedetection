function [theta,data,p_C_hat] = simulate_data(model,condition)

data.pres2stimuli = condition;
nDelta = 20;
delta = [linspace(0,pi,nDelta)' ones(nDelta,3)];
rel = [0.5 0.5 0.5 0.5;
       0.9 0.5 0.5 0.5;
       0.5 0.5 0.5 0.9;
       0.9 0.9 0.5 0.5;
       0.5 0.5 0.9 0.9;
       0.9 0.9 0.9 0.5;
       0.5 0.9 0.9 0.9;
       0.9 0.9 0.9 0.9];

% get settings for current model
[logflag,~,~,PLB,PUB] = getFittingSettings(model, condition);

% get simulated theta
nParams = length(PLB);
theta = (PUB-PLB).*rand(1,nParams)+PLB;

% calculate p_C_hat
[~,p_C_hat] = calculate_LL(theta,data,model,logflag,nSamples);