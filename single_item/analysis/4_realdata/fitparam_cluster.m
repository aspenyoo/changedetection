function fitparam_cluster(jobnum, nStartVals)
%FITPARAM_REALDATA(SUBJIDS,MODEL,NSTARTVALS) fits the parameters using
%JOBNUM shouls be a number where the model is first digit (tens or
%hundreds) and the subject number is next (ones or tens and ones if the
%subjnumber is larger than 
%NSTARTVALS starting values for model MODEL and subjects with IDs SUBJIDS.
%data is saved in a MATLAB file
%
if nargin < 3; nStartVals = 10; end

jobnum = num2str(jobnum);
subjids = {'1','3','4','ND','EL'};
model = str2num(jobnum(1));
subjid = subjids{str2num(jobnum(2:end))};

% just loading the first subject to see how many conditions there are
[data] = concatdata(subjid,'Detection');
[Xdet] = conditionSeparator(data);
nCond = length(Xdet);

switch model
    case 1 % optimal model (noise param per cond., lapse)
        nParams = nCond + 1;
    case 2 % optimal model + p(common)
        nParams = nCond + 2;
    case 3 % optimal model + p(common) + p(S)
        nParams = nCond + 3;
    case 5 % fixed criterion model (noise param per cond, 1 internal noise, lapse)
        nParams = nCond + 2;
    case 6 % superfree model (true and internal noise param per cond, lapse)
        nParams = nCond*2 + 1;
end
formatSpec = repmat('%4.4f \t ',1,nParams+2);
formatSpec = [formatSpec(1:end-3) '\r\n'];
permission = 'a+';

[bestFitParam, nLL_est] = fitparam(Xdet, model, nStartVals);

c = clock;
filename = sprintf('paramfit_model%d_subj%s_%02d%02d%d.txt',model,upper(subjid),c(2),c(3),c(1));
fileID = fopen(filename,permission);
A1 = [bestFitParam, nLL_est, nStartVals];
fprintf(fileID, formatSpec, A1); % save stuff in txt file
fclose(fileID);



