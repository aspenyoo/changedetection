function [bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples)
% %FIT_PARAMETERS was made by aspen oct 31 2018 to try to investigate
% %different aspects of model. same as fit_maximum_likelihood but outputs
% %varaibles instead of saving them to a file

if nargin < 5 || isempty(runmax); runmax = 50; end
if nargin < 6 || isempty(nSamples); nSamples = 1e4; end

% # samples for high-precision estimate
if numel(nSamples) > 1
    nSamplesFinal = nSamples(2);
else
    nSamplesFinal = 4e5;
end


% =====================================================================
%                     MODEL + CONDITION STUFF
% =====================================================================
% model indices
encoding = model(1);        % actual noise. 1: VP, 2: FP
variability = model(2);     % assumed noise. 1: VP, 2: FP, 3: single value
decision_rule = model(3);   % decision rule. 1: optimal, 2: max
condition = data.pres2stimuli;
% decision_noise = model(4);  % decision noise: 1: none, 2: local, 3: global

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

% BPS options
options.UncertaintyHandling = 'on';

% Generate set of starting point with a Latin hypercube design
rng(0); % Same set for all
nvars = numel(PLB);
x0_list = lhs(runmax,nvars,PLB,PUB,[],1e3);

filename = sprintf('fits/subj%s_%s_model%d%d%d.mat',data.subjid,condition,model(1),model(2),model(3));
if exist(sprintf('fits/%s',filename),'file')
else
    [bfp, LLVec, completedruns] = deal([]);
end


for iter = 1:numel(runlist)
    fprintf('iteration number: %d \n',runlist(iter))
    tic;
    
    % Fix random seed based on iteration (for reproducibility)
    rng(runlist(iter));
    
    x0 = x0_list(runlist(iter),:);
    [xbest,LL,~,~] = ...
        bads(@(x) -calculate_LL(x,data,model,logflag,nSamples),x0,LB,UB,PLB,PUB,[],options);

    xbest(logflag) = exp(xbest(logflag)); % getting parameters back into natural units
    bfp = [bfp; xbest];
    LLVec = [LLVec; LL];
    completedruns = [completedruns; runlist(iter)];
    save(filename,'bfp','LLVec','completedruns')
    toc
end

end
