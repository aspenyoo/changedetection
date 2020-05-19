function [bfp, LLVec, completedruns] = find_ML_parameters(data,model,runlist,runmax,nSamples,useprevfits)
% % FIT_PARAMETERS was made by aspen oct 31 2018 to try to investigate
% % different aspects of model. same as fit_maximum_likelihood but outputs
% % varaibles instead of saving them to a file

if nargin < 4 || isempty(runmax); runmax = 50; end
if nargin < 6; useprevfits = 0; end

% get model fitting settings
condition = data.pres2stimuli;
[logflag,LB,UB,PLB,PUB] = getFittingSettings(model, condition);

% BADS options
options.UncertaintyHandling = 'on';

% Generate set of starting point with a Latin hypercube design
if ~sum(useprevfits)
    rng(0); % Same set for all
    nvars = numel(PLB);
    x0_list = lhs(runmax,nvars,PLB,PUB,[],1e3);
end

filename = sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,data.subjid,condition,model(1),model(2),model(3),model(4));
filename

% ibs bads settings
dMat = data.Delta;
rels = unique(data.rel);
blah = data.rel;
for irel = 1:length(rels)
    blah(blah == rels(irel)) = irel;
end
dMat = [dMat blah];

for iter = 1:numel(runlist)
    fprintf('iteration number: %d \n',runlist(iter))
    
    % Fix random seed based on iteration (for reproducibility)
    rng(runlist(iter));
    
    % ibs bad settings
    options_ibs = ibslike('defaults');
    options_ibs.Vectorized = 'on';
    options_ibs.MaxIter = 10000;
%     options_ibs.NegLogLikeThreshold = 20000;%Inf;%log(2)*2000; % log(2)nTrials

    if sum(useprevfits)
        x0 = useprevfits;
        x0(logflag) = log(x0(logflag))
        LB
        UB
        LB < x0
        UB > x0
    else
        x0 = x0_list(runlist(iter),:);
        x0 = x0+rand(size(x0))*0.1-0.05; % adding a tiny bit of noise for jobs that are not converging
        x0 = min(max(x0,LB+1e-3),UB-1e-3); % make sure it is within bounds
    end
    
%     [xbest,LL,~,~] = ...
%             bads(@(x) -calculate_LL(x,data,model,logflag,nSamples(1)),x0,LB,UB,PLB,PUB,[],options)

    % this is for ibs bads
    
        fun = @(x,y) fun_LL(x,y,model,condition,logflag);
    [xbest,LL,~,~] = bads(@(x) ibslike(fun,x,data.resp,dMat,options_ibs),x0,LB,UB,PLB,PUB,[],options)
    
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
