function fitparam_realdata(subjids,model, nStartVals)
%FITPARAM_REALDATA(SUBJIDS,MODEL,NSTARTVALS) fits the parameters using
%NSTARTVALS starting values for model MODEL and subjects with IDs SUBJIDS.
%data is saved in a MATLAB file
%
if nargin < 3; nStartVals = 10; end

nSubj = length(subjids);
filepath = 'changedetection/analysis/4_realdata/fits/';

nCond = 5; % number of conditions in current version of the experiment (052016)
switch model
    case 1 % optimal model (noise param per cond., lapse)
        nParams = nCond + 1;
    case 2 % optimal model + p(common)
        nParams = nCond + 2;
    case 3 % optimal model + p(common) + p(S)
        nParams = nCond + 3;
    case 4 % optimal model + p(S)
        nParams = nCond + 2;
    case 5 % fixed criterion model (noise param per cond, 1 internal noise, lapse)
        nParams = nCond + 2;
    case 6 % superfree model (true and internal noise param per cond, lapse)
        nParams = nCond*2 + 1;
    case 7 % linear heuristic model (noise + 2 kcommon parameters + lapse)
        nParams = nCond + 3;
    case 11 % optimal w/ power law. power law mapping for kappas, lapse
        nParams = 3 + 1;
    case 15 % fixed w/ power law. power law mapping for kappas, lapse, assumed kappa
        nParams = 3 + 2;
    case 17 % heuristic w/ power law
        nParams = 3 + 3;
        
end
formatSpec = repmat('%4.4f \t ',1,nParams+1);
formatSpec = [formatSpec(1:end-3) '\r\n'];
permission = 'a+';

% bestFitParam = nan(nSubj,nParams);
% nLL_est = nan(1,nSubj);
for isubj = 1:nSubj;
    subjid = subjids{isubj}
    
    % getting data
    [data] = concatdata(subjid,'Detection');
    [Xdet] = conditionSeparator(data);
    
    % setting up stuff for saving MLE parameter estimates
    filename = sprintf('%sparamfit_model%d_subj%s.txt',filepath,model,upper(subjid));

    % check how many jobs have been completed
    try
        count = size(dlmread(filename),1); % how many jobs were completed
        nStartVal = nStartVals - count; % how many jobs you have to do to get to nStartVals total
    catch
        nStartVal = nStartVals;
    end
    
    for istartval = 1:nStartVal;
        [bestFitParam, nLL_est] = fitparam(Xdet, model);
        
        % open, save, and close for each iteration
        fileID = fopen(filename,permission);
        A1 = [bestFitParam, nLL_est];
        fprintf(fileID, formatSpec, A1); % save stuff in txt file
        fclose(fileID);
    end
   
end

