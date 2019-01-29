function settings = getExperimentSettings()

deltavartyperange = []; % range of possible deltas
reliabilitytype=''; % mixed or uniform eccentricity (reliably) ellipses
reliabilityval=[]; % value(s) of low eccentricity ellipses
setsizeval=[]; % set size
nTrials = []; % number of trials
breaknum =[]; % number of breaks
stim_on_time = []; % stimulus on time (per display)
expID = ''; % experiment type
dir_names = dir('./output');

% Get subject ID
subjid   = input('Subject initials......................................: ','s');
dir_present = 0;
for i = 1:size(dir_names,1)
    if strcmp(subjid,dir_names(i).name)
        dir_present = 1;
    end
end

if ~dir_present
    mkdir(['./output/' subjid]);
end

% If this is an existing experimental setup, use those parameters
while ~(strcmp(expID,'Reliability') || strcmp(expID,'Threshold') || strcmp(expID,'Practice'))
    expID = input('Experiment ID (Reliability/Threshold/Practice)............: ','s');
end

try
    load(['./output/' subjid '/low_rel'])
    if strcmp(expID,'Threshold')
        fprintf('\nThreshold Set! Continue?\n')
        redo_thresh = input('Y/N:','s');
        if ~strcmp(redo_thresh,'Y') && ~strcmp(redo_thresh,'y')
            error('Threshold was already set!')
        end
    end
catch    
    if strcmp(expID,'Reliability')
        error('Threshold not set!')
    end 
end


if(strcmp(expID,'Reliability'))

    % Set parameters for the experimental trials
    deltavartyperange = 180;
    reliabilitytype='mixed';
    reliabilityval= [low_rel .9];
    setsizeval= 4;    
    nTrials = 800;
    breaknum = 4;
    stim_on_time = 0.1;
    
elseif(strcmp(expID,'Threshold'))

    % Set parameters for the threshold trials
    deltavartyperange = 180;
    reliabilitytype='constant';
    reliabilityval= linspace(.3,.9,16);
    setsizeval= 4;    
    nTrials = 400;
    breaknum = 2;
    stim_on_time = 0.1;

elseif(strcmp(expID,'Practice'))

    % Set parameters for the practice trials    
    deltavartyperange = 180;
    reliabilitytype='constant';
    reliabilityval= linspace(.3,.9,16);
    setsizeval= 4;    
    nTrials = 256;
    breaknum = 1;
    stim_on_time = 0.333;

end

settings.subjid = upper(subjid);
settings.deltavartyperange = deltavartyperange;
settings.reliabilitytype = reliabilitytype;
settings.reliabilityval = reliabilityval;
settings.expID = expID;
settings.setsizeval = setsizeval;
settings.breaknum = breaknum;
settings.nTrials = nTrials;
settings.stim_on_time = stim_on_time;