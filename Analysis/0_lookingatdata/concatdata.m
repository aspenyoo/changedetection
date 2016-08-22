function [designMat, stimuliMat] = concatdata(subjid, experimenttype, sessionnumbers)
if nargin < 2; experimenttype = 'Discrim'; end
if nargin < 3; sessionnumbers = []; end
if nargout < 1; issave = 1; else issave = 0; end

subjid = char(subjid);
filepath = 'Experiment and Data/output_mat';
title = ['Exp_Change' experimenttype '_Simult'];
concatfilename = [title '_concat_subj' subjid '.mat'];

% check if concatinated data already exists
files = dir([filepath '/' concatfilename]);

if ~isempty(files)
    load(files.name);
else    
    if isempty(sessionnumbers);
        % load all data
        files = dir([filepath '/' title '_subj' subjid '*.mat']);
        if isempty(files);
            error('no files with that subjid in current directory')
        end
        
        % delete training session numbers (mod(session,11) = 0)
        idx = [];
        for ifile = 1:length(files);
            if files(ifile).name(40+length(subjid)) == files(ifile).name(41+length(subjid))
                idx = [idx ifile];
            end
        end
        files(idx) = [];
    else
        % load data from specified sessionnumbers
        nFiles = length(sessionnumbers);
        
        j = 1;
        for i = 1:nFiles
            sessionnum = sessionnumbers{i};
            filestuff = dir([filepath '/' title '_subj' subjid ...
                '_session' sessionnum '_*.mat']);
            for ii = 1:length(filestuff);
                files(j).name = filestuff(ii).name;
                j = j+1;
            end
        end
    end
    
    dMat = []; sMat = [];
    for i=1:length(files);
        load(files(i).name);
        stimuliMat(isnan(designMat(:,end)),:) = [];
        designMat(isnan(designMat(:,end)),:) = [];
        dMat = [dMat; designMat];
        sMat = [sMat; stimuliMat];
    end
    
    designMat = dMat;
    stimuliMat = sMat;
    
    
    if (issave)
        save([filepath '/' concatfilename],'designMat','stimuliMat','names');
    end
    
end

