function [bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples)
% %FIT_PARAMETERS was made by aspen oct 31 2018 to try to investigate
% %different aspects of model. same as fit_maximum_likelihood but outputs
% %varaibles instead of saving them to a file

if nargin < 4 || isempty(runmax); runmax = 50; end
if nargin < 5 || isempty(nSamples); nSamples = [50 1000]; end

assert(length(nSamples) == 2, 'length of nSamples must be 2')

% % # samples for high-precision estimate
% if numel(nSamples) > 1
%     nSamplesFinal = nSamples(2);
% else
%     nSamplesFinal = 2000;
% end

% get model fitting settings
condition = data.pres2stimuli;
[logflag,LB,UB,PLB,PUB] = getFittingSettings(model, condition);

% BPS options
options.UncertaintyHandling = 'on';

% Generate set of starting point with a Latin hypercube design
rng(0); % Same set for all
nvars = numel(PLB);
x0_list = lhs(runmax,nvars,PLB,PUB,[],1e3);

filename = sprintf('fits/%s/subj%s_%s_model%d%d%d.mat',condition,data.subjid,condition,model(1),model(2),model(3));
filename

for iter = 1:numel(runlist)
    fprintf('iteration number: %d \n',runlist(iter))
    
    % Fix random seed based on iteration (for reproducibility)
    rng(runlist(iter));
    
    x0 = x0_list(runlist(iter),:);
    [xbest,~,~,~] = ...
        bads(@(x) -calculate_LL(x,data,model,logflag,nSamples(1)),x0,LB,UB,PLB,PUB,[],options)

    % recalculate LL with more samples
    LL = -calculate_LL(xbest,data,model,logflag,nSamples(2));
    
    xbest(logflag) = exp(xbest(logflag)); % getting parameters back into natural units
    
    % it is necessary to reload the file at every iteration in case multiple processors are
    % saving the file at the same time
    if exist(filename,'file')
        load(filename,'bfp','LLVec','completedruns')
    else
        [bfp, LLVec, completedruns] = deal([]);
    end
    
    
    
    % update and save variables
    bfp = [bfp; xbest];
    LLVec = [LLVec; LL];
    completedruns = [completedruns; runlist(iter)];
    save(filename,'bfp','LLVec','completedruns')
end

end
