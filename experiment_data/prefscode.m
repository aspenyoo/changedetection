function prefs = prefscode(afctype, exptype, subjid, sessionnum, nTrialsPerCond)

if nargin < 3; subjid = []; end
if nargin < 4; sessionnum = 0; end
if nargin < 5; nTrialsPerCond = []; end

if nargin < 1; afctype = input('Discrim/Detect: ','s');end
if nargin < 2; exptype = input('Simult/Seq: ' , 's'); end
if isempty(subjid); subjid = input('enter 3 letter subject ID: ', 's'); end
if isempty(nTrialsPerCond);
    if strcmp(afctype,'Discrim'); nTrialsPerCond = 30;end
    if strcmp(afctype,'Detect'); nTrialsPerCond = 150; end
end

subjid = upper(subjid);
dateStr = datestr(now, 30);

%=========================================================================
% THINGS THAT ALL EXPERIMENTS SHOULD HAVE IN COMMON
% ========================================================================

% colors
red                   = [200 0 0];
green                 = [0 200 0];
white                 = 255;
lightgrey             = 200;  % foreground
grey                  = 128;  % background
black                 = 0;

% experiment timing (sec)
fix1Dur = 0.5;  % fixation screen in the beginning
pres1Dur = .09;      % presentation 1
pres1ISIDur = 1;    % inter-stimulus interval w/in pres1 (only for seq)
pres2Dur = .100;        % will not exist when respInPres2 == 1
ITIDur = 0.5;          % inter-trial interval

% display settings
screenHeight    = 29.5;   % in cm (Dell@T115A: ~48cm; Dell@T101C: ~40 cm)
bgColor         = grey; % background color
stimColor       = lightgrey;  % stimulus color
fixLength       = 7;          % fixation cross length
fixColor        = black;      % fixation cross color
jitter          = 0;   % amount of x/y-jitter (deg)
stimecc         = 7;    % stimulus eccentricity (deg)
ellipseArea     = 1;   % ellipse area (deg)

% ========================================================================
% EXPERIMENT SPECIFIC
% ========================================================================

if strcmp(afctype(end-5:end),'Detect'); % detection task (yes/no)
    
    % info for current experiment
    expName = ['Exp_ChangeDetection_' exptype '_subj' num2str(subjid) '_session' num2str(sessionnum)];
    
    if strcmp(exptype,'Simult'); % simultaneous detection
        
        % yes(1)/no(0)
        vmprior = 8.742;      % kappa of vm prior. 0 if uniform prior. (used to be sd of gauss prior)
        screenshot = 0;      % do you want a screenshot?
        feedback = 0;        % do you want to give feedback on performance?
        allStimInPres2 = 0;  % all stimuli to be in pres2 (vs.just target)?
        directionChange = 0; % task clockwise/counterclockwise (vs. yes/no)?
        respInPres2 = 1;     % does 2nd presentation stay up until S responds?
        simultPres1 = 1;        % are all stimuli presented simultaneous in first presentation?
        permLocInPres1 = 0;  % are stimuli locations in pres1 permuted?
        
        % breaks and feedback
        blocknum = 5;     % number of blocks ( 1 + number of breaks )
        breakDuration = 20;   % duration of each break between blocks (sec)
        feedbacktrial = 29; % every feedbacktrialth trial, feedback will be given
        
        % experimental design
        deltaNum = [0 1]; %
        f1 = length(deltaNum);
        % if directionChange = 1, magnitude and direction of change
        % if directionChange = 0, should be 0 or 1 for yes/no change
        
        setSizeNum =[1]; % stimulus set size
        f2 = length(setSizeNum);
        
        %         reliabilityName = {'low','medium', 'high'};
        reliabilityNum = [0.15 0.3 0.5 0.8 0.999]; % ellipse eccentricity
        f3 = length(reliabilityNum);
        
        ISIdelayNum = [0.981];   % ISI delay time (sec)
        f4 = length(ISIdelayNum);
        
    end
elseif strcmp(afctype,'Discrim'); % discrimination task (c/cc)
    
    % info for current experiment
    expName = ['Exp_ChangeDiscrim_' exptype '_subj' num2str(subjid) '_session' num2str(sessionnum)];

    
    % yes(1)/no(0)
    gaussprior = 15;      % gaussian or stepwise
    screenshot = 0;      % do you want a screenshot?
    feedback = 1;        % do you want to give feedback on performance?
    allStimInPres2 = 0;  % all stimuli to be in pres2 (vs.just target)?
    directionChange = 1; % task clockwise/counterclockwise (vs. yes/no)?
    respInPres2 = 1;     % does 2nd presentation stay up until S responds?
    simultPres1 = 1;        % are all stimuli presented simultaneous in first presentation?
    permLocInPres1 = 0;  % are stimuli locations in pres1 permuted?
    
    
    % breaks and feedback
    blocknum = 1;     % number of blocks ( 1 + number of breaks )
    breakDuration = 20;   % duration of each break between blocks (sec)
    feedbacktrial = []; % every feedbacktrialth trial, feedback will be given
    
    
    % experimental design
    directionNum = [1]; % always changing
    f1 = length(directionNum);
    
    setSizeNum = [1]; % stimulus set size
    f2 = length(setSizeNum);
    
    
    ISIdelayNum = [0.981]; % [.5 1 2];   % ISI delay time (sec)
    f4 = length(ISIdelayNum);
    
    
    if strcmp(exptype,'Simult'); % simultaneous discrim
        
        % reliabilityName = ['low', 'medium', 'high'];
        reliabilityNum = [0.6 0.95 0.999]; % ellipse eccentricity [.65 .97];%
        f3 = length(reliabilityNum);
        
    elseif strcmp(exptype,'sausage') % distrimination sausages sumlationously
        
        % reliabilityName = ['gabor', 'lilsausage', 'bigsausage'];
        reliabilityNum = [0 pi/4 pi/2]; % % sausage length
        f3 = length(reliabilityNum);
        
    end
    
    % elseif strcmp(afctype,'Train1'); % discrimination task (c/cc)
    %     reliabilityNum = [0.6 0.95 0.999]; % ellipse eccentricity [.65 .97];%
end


if strcmp(afctype(1:5),'Pract') % detection task (yes/no)
    
    % info for current experiment
    expName = ['Pract' expName(4:end)]; % changing the name to have "pract" instead of "exp"
    
    feedback = 1;        % feedback
    pres1Dur = pres1Dur*2;      % twice as long ellipse presentation time
    blocknum = 1;     % number of blocks ( 1 + number of breaks )
end

if sum(setSizeNum == 1)==1;
    stimecc = 0;
    jitter = 0;
end

% making struct
varNames = who;
fieldNames1 = ['fieldNames' varNames'];
prefs = v2struct(fieldNames1);


