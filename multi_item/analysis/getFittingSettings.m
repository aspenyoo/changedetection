function [LB,UB,PLB,PUB,logflag] = getFittingSettings(model, condition)

% model indices
encoding = model(1);        % actual noise. 1: VP, 2: FP
variability = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max

% Set parameter bounds
jbar_bounds = [0.0067 50];  % Hard bounds for JBAR1 and JBAR2
jbar_pbounds = [1 25];  % Plausible bounds for JBAR1 and JBAR2
tau_bounds = [1e-3 100];     % Hard bounds for TAU
tau_pbounds = [0.5 30];     % Plausible bounds for TAU
crit_bounds = [-10 20];
crit_pbounds = [-2 2];  
pchange_bounds = [1e-4 1];
pchange_pbounds = [0.3 0.6];

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

switch decision_rule
    case 1 % if optimal, need prior over p(change)
        
        if (variability == 3) % if participant believes they have one noise for ellipse (and one for line)
            LB = [LB jbar_bounds(1)];
            UB = [UB jbar_bounds(2)];
            PLB = [PLB jbar_pbounds(1)];
            PUB = [PUB jbar_pbounds(2)];
            logflag = [logflag 1];
            
            % if Line condition, need an additional Jbar value for assumed Jbar
            if strcmp(condition,'Line')
                LB = [LB jbar_bounds(1)];
                UB = [UB jbar_bounds(2)];
                PLB = [PLB jbar_pbounds(1)];
                PUB = [PUB jbar_pbounds(2)];
                logflag = [logflag 1];
            end
        end
        
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