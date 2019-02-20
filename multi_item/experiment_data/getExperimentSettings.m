function settings = getExperimentSettings()

deltavartyperange = []; % range of possible deltas
reliabilitytype=''; % mixed or uniform eccentricity (reliably) ellipses
reliabilityval=[]; % value(s) of low eccentricity ellipses
setsizeval=[]; % set size
nTrials = []; % number of trials
breaknum =[]; % number of breaks
stim_on_time = []; % stimulus on time (per display)
expID = ''; % experiment type
pres2stimuli = ''; % shape of stimuli in second presentaion
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
while ~(strcmp(expID,'R') || strcmp(expID,'T') || strcmp(expID,'P') || strcmp(expID,'R2'))
    expID = input('Experiment ID (Reliability/Threshold/Practice)............: ','s');
end
while ~(strcmp(pres2stimuli,'E') || strcmp(pres2stimuli,'L'))
    pres2stimuli = input('Stimuli in second presentation (Ellipse/Line)............: ','s');
end
if (strcmp(expID,'P')); expID = 'Practice'; end
if (strcmp(expID,'T')); expID = 'Threshold'; end
if (strcmp(expID,'R')); expID = 'Reliability'; end
if (strcmp(expID,'R2')); expID = 'Reliability2'; end
if (strcmp(pres2stimuli,'E')); pres2stimuli = 'Ellipse'; end
if (strcmp(pres2stimuli,'L')); pres2stimuli = 'Line'; end

try
    load(['./output/' subjid '/lowrel_' pres2stimuli])
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

% set some condition-independent variables
settings.makeScreenShot  = 0;      % if 1, then Screenshots of stimuli will be made
settings.ScreenHeight    = 28.5;   % malab experimental room 758 % in cm (Dell@T115A: ~48cm; Dell@T101C: ~40 cm)
settings.ellipseArea     = 1.19;   %settings.barwidth*settings.barheight; % ellipse size (deg^2)
% the linewidth and length are set to approximate the ellipse area!
settings.lineWidth       = 6;      % width of line, for presentation 2 (pixels)
settings.lineLength      = 135;    % length of line for pres 2 simuli (pixels)
% settings.barwidth        = 0.3;  % width of stimulus bar (deg)
% settings.barheight       = 0.8;  % height of stimulus bar (deg)
settings.jitter          = 0.6;    % amount of x/y-jitter (deg)
settings.bgdac           = 128;    % background grayvalue (RGB)
settings.fgdac           = 200;    % foreground grayvalue (RGB)
settings.stimecc         = 7;      % stimulus eccentricity (deg)
settings.ITI             = 1;      % inter stimulus time (sec)
settings.breaktime       = 10;     % mandatory breaktime (sec)

switch expID
    case 'Practice' % Practice
        
        % Set parameters for the practice trials
        deltavartyperange = 180;
        reliabilitytype='constant';
        reliabilityval= linspace(.3,.9,16);
        setsizeval= 4;
        nTrials = 256;
        breaknum = 2;
        stim_on_time = 0.333;
        feedback = 1;
            
    case 'Threshold' % Threshold
        
        % Set parameters for the threshold trials
        deltavartyperange = 180;
        reliabilitytype='constant';
        reliabilityval= linspace(.3,.9,16);
        setsizeval= 4;
        nTrials = 400;
        breaknum = 4;
        stim_on_time = 0.1;
        feedback = 1;
        
    case 'Reliability' % Reliability
        
        % Set parameters for the experimental trials
        deltavartyperange = 180;
        reliabilitytype='mixed';
        reliabilityval= [low_rel .9];
        setsizeval= 4;
        nTrials = 900;
        breaknum = 8;
        stim_on_time = 0.1;
        feedback    = 0;    % feedback flag

    case 'Reliability2' % Reliability (short version)
        
        % Set parameters for the experimental trials
        deltavartyperange = 180;
        reliabilitytype='mixed';
        reliabilityval= [low_rel .9];
        setsizeval= 4;
        nTrials = 100;
        breaknum = 0;
        stim_on_time = 0.1;
        feedback    = 0;    % feedback flag
    
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
settings.feedback = feedback;
settings.pres2stimuli = pres2stimuli;