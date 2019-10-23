function [logflag,LB,UB,PLB,PUB] = getFittingSettings(model, condition)


% CURRENT: PARAMETER VALUES FOR DIFFERENT MODELS
% [1 1 1], [1 1 2], [1 2 1], [1 2 2]: 
%   [Jbar_high, Jbar_low,( Jbar_line,) tau, p_change/criterion]
% [1 3 1]:
%   [Jbar_high, Jbar_low,( Jbar_line,) tau, Jbar_assumedellipse, p_change/criterion]

% OLD: PARAMETER VALUES FOR DIFFERENT MODELS
% [1 1 1], [1 1 2], [1 2 1], [1 2 2]: 
%   [Jbar_high, Jbar_low,( Jbar_line,) tau, p_change/criterion]
% [1 3 1]:
%   [Jbar_high, Jbar_low,( Jbar_line,) tau, Jbar_assumedellipse,( Jbar_assumedline,) p_change/criterion]

% model indices
encoding = model(1);        % actual noise. 1: VP, 2: FP
infering = model(2);        % assumed noise. 1: VP, 2: FP, 3: diss assumed for ellipse and line, 4: ignore completely
decision_rule = model(3);   % decision rule. 1: optimal, 2: max
decision_noise = model(4);  % decision noise: 0: none, 1: local, 2: global

% Set parameter bounds
jbar_bounds = [1e-4 50];        % Hard bounds for JBAR1 and JBAR2
jbar_pbounds = [1e-4 10];       % Plausible bounds for JBAR1 and JBAR2
tau_bounds = [1e-3 100];        % Hard bounds for TAU
tau_pbounds = [0.5 30];         % Plausible bounds for TAU
crit_bounds = [-1e3 1e3];
crit_pbounds = [-10 10];  
pchange_bounds = [1e-4 1];
pchange_pbounds = [0.3 0.6];
sigma_d_bounds = [1e-4 50];
sigma_d_pbounds = [0.5 20];

% set Jbar_high and Jbar_low bounds
LB = [jbar_bounds(1) jbar_bounds(1)]; 
UB = [jbar_bounds(2) jbar_bounds(2)];
PLB = [jbar_pbounds(1) jbar_pbounds(1)];
PUB = [jbar_pbounds(2) jbar_pbounds(2)];
logflag = [1 1];

% if Line condition, need an additional Jbar value for actual encoding
% precision
if strcmp(condition,'Line')
    LB = [LB jbar_bounds(1)];
    UB = [UB jbar_bounds(2)];
    PLB = [PLB jbar_pbounds(1)];
    PUB = [PUB jbar_pbounds(2)];
    logflag = [logflag 1];
end

% if VP, add tau parameter
if (encoding == 1); 
    LB = [LB tau_bounds(1)];
    UB = [UB tau_bounds(2)];
    PLB = [PLB tau_pbounds(1)];
    PUB = [PUB tau_pbounds(2)];
    logflag = [logflag 1];
end

if (infering >= 3) && ((infering==4) && (decision_rule==2))% if participant believes they have one noise for ellipse (4) (and potentiall one for line; 3)
    LB = [LB jbar_bounds(1)];
    UB = [UB jbar_bounds(2)];
    PLB = [PLB jbar_pbounds(1)];
    PUB = [PUB jbar_pbounds(2)];
    logflag = [logflag 1];
    
    % if Line condition and Same variability, need an additional Jbar value for assumed Jbar
    if strcmp(condition,'Line') && (infering == 3)
        LB = [LB jbar_bounds(1)];
        UB = [UB jbar_bounds(2)];
        PLB = [PLB jbar_pbounds(1)];
        PUB = [PUB jbar_pbounds(2)];
        logflag = [logflag 1];
    end
end

if (decision_noise)
    % sigma_d
    LB = [LB sigma_d_bounds(1)];
    UB = [UB sigma_d_bounds(2)];
    PLB = [PLB sigma_d_pbounds(1)];
    PUB = [PUB sigma_d_pbounds(2)];
    logflag = [logflag 1];
end

switch decision_rule
    
    case 1 % if optimal, need prior over p(change)
        % p(change)
        LB = [LB pchange_bounds(1)];
        UB = [UB pchange_bounds(2)];
        PLB = [PLB pchange_pbounds(1)];
        PUB = [PUB pchange_pbounds(2)];
        logflag = [logflag 0];
        
    case 2 % if max, need criterion
        LB = [LB crit_bounds(1)];
        UB = [UB crit_bounds(2)];
        PLB = [PLB crit_pbounds(1)];
        PUB = [PUB crit_pbounds(2)];
        logflag = [logflag 0];
end

% logging the relevant ones
logflag = logical(logflag);
LB(logflag) = log(LB(logflag));
UB(logflag) = log(UB(logflag));
PLB(logflag) = log(PLB(logflag));
PUB(logflag) = log(PUB(logflag));