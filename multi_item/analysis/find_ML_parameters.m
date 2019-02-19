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
    nSamplesFinal = 2000;
end

% get model fitting settings
condition = data.pres2stimuli;
[LB,UB,PLB,PUB,logflag] = getFittingSettings(model, condition);

% BPS options
options.UncertaintyHandling = 'on';

% Generate set of starting point with a Latin hypercube design
rng(0); % Same set for all
nvars = numel(PLB);
x0_list = lhs(runmax,nvars,PLB,PUB,[],1e3);

filename = sprintf('fits/subj%s_%s_model%d%d%d.mat',data.subjid,condition,model(1),model(2),model(3));

for iter = 1:numel(runlist)
    fprintf('iteration number: %d \n',runlist(iter))
    
    % Fix random seed based on iteration (for reproducibility)
    rng(runlist(iter));
    
    x0 = x0_list(runlist(iter),:);
    [xbest,~,~,~] = ...
        bads(@(x) -calculate_LL(x,data,model,logflag,nSamples),x0,LB,UB,PLB,PUB,[],options);
    LL = -calculate_LL(xbest,data,model,logflag,nSamplesFinal);
    
    xbest(logflag) = exp(xbest(logflag)); % getting parameters back into natural units
    
    % it is necessary to reload the file at every iteration in case multiple processors are
    % saving the file at the same time
    if exist(sprintf('fits/%s',filename),'file')
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
