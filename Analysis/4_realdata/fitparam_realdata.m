function fitparam_realdata(subjids,model, nStartVals)
%FITPARAM_REALDATA(SUBJIDS,MODEL,NSTARTVALS) fits the parameters using
%NSTARTVALS starting values for model MODEL and subjects with IDs SUBJIDS.
%data is saved in a MATLAB file
%
if nargin < 3; nStartVals = 10; end

nSubj = length(subjids);

% just loading the first subject to see how many conditions there are
% [data] = concatdata(subjids{1},'Detection');
% [Xdet] = conditionSeparator(data);
% nCond = length(Xdet);

nCond = 5; % in current version of the experiment
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
end
formatSpec = repmat('%4.4f \t ',1,nParams+2);
formatSpec = [formatSpec(1:end-3) '\r\n'];
permission = 'a+';

% bestFitParam = nan(nSubj,nParams);
% nLL_est = nan(1,nSubj);
parfor isubj = 1:nSubj;
    subjid = subjids{isubj};
%     if subjid(1) == 'F' % if fake data
%         load(['fakedata_model' subjid(2) '_subjF' subjid(3:end) '.mat'])
%     else
        [data] = concatdata(subjid,'Detection');
        [Xdet] = conditionSeparator(data);
%     end
    
    %     [bestFitParam(isubj,:), nLL_est(isubj)] = fitparam(Xdet, model, nStartVals);
    [bestFitParam, nLL_est] = fitparam(Xdet, model, nStartVals);
    
    c = clock;
    filename = sprintf('paramfit_model%d_subj%s_%02d%02d%d.txt',model,upper(subjid),c(2),c(3),c(1));
    fileID = fopen(filename,permission);
    A1 = [bestFitParam, nLL_est, nStartVals];
    fprintf(fileID, formatSpec, A1); % save stuff in txt file
    fclose(fileID);
    
end

