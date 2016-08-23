function [bestFitParam, nLL_est, subjids] = combineparamfit(model,filepath)
if nargin < 2; filepath = 'Analysis/4_realdata/fitparam'; end
% combines data from cluster (txt files) for model
%
% NOTES
% assumption is that each subject only has best fit parameter in the txt
%   file
% txt file should be set up [bfp nll (other stuff I ignore)]
% 
% aspen.yoo
% April 18, 2016

nCond = 5;
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
    case 7 % linear heuristic model (true noise param per condition, kcommon low and high, lapse)
        nParams = nCond + 3;
end



filename = ['paramfit_model' num2str(model) '_subj'];

files = dir([filepath '/' filename '*.txt']);
nFiles = length(files); % = nSubj

bestFitParam = nan(nFiles,nParams);
nLL_est = nan(1,nFiles);
subjids = cell(1,nFiles);
for ifile = 1:nFiles;
    files(ifile).name
    data = dlmread(files(ifile).name);
    bestFitParam(ifile,:) = data(1:nParams);
    nLL_est(ifile) = data(nParams+1);
    subjids{ifile} = files(ifile).name(length(filename)+1:find(files(ifile).name == '_',1,'last')-1);
end

if nargout < 1;
    c = clock;
    save([filepath '/' sprintf('paramfit_model%d_%02.0f%02.0f%04.0f.mat',...
        model,c(2),c(3),c(1))],'bestFitParam','nLL_est','subjids');
end