function Exp_OrientationChangeDiscrim(subjid, sessionnum, nTrialsPerCond)
% runs experiment: orientation change discrimination
%
% STRUCT EXPLANATIONS
% prefs: experimental preferences
% D: related to data (things you may analyze about the experiment in the
% future)
%
% last updated Sept 2, 2015 by Aspen Yoo (aspen.yoo@nyu.edu)

if nargin < 1; subjid = []; end
if nargin < 2; sessionnum = []; end
if nargin < 3; nTrialsPerCond = []; end

commandwindow;
% ========================================================================
% SETTING THINGS UP (changing this section is usually unnecessary)
% ========================================================================

% % clearing/closing everything
% close all;
% sca;

savedata = 1;

% random number generator
rng('shuffle')

% skipping sync tests
% Screen('Preference', 'SkipSyncTests', 1);

% ========================================================================
% PREFERENCES (should be changed to match experiment preferences)
% ========================================================================


% importing preferences for simultaneous/sequential experiment
prefs = prefscode('Discrim','Simult',subjid, sessionnum, nTrialsPerCond);

% full factorial design
prefs.design = fullfact([prefs.f1 prefs.f2 prefs.f3 prefs.f4]);
prefs.nCond = size(prefs.design,1);
prefs.conditionNum = [1:prefs.nCond]';
prefs.design = [prefs.design prefs.conditionNum];

% testing design (trials/condition, pseudo-randomizing order)
% prefs.nTrialsPerCond = nTrialsPerCond;
% prefs.nTrialsPerCond = 100;       % trials per condition
prefs.fullDesign = repmat(prefs.design,[prefs.nTrialsPerCond 1]);
prefs.nTrials = size(prefs.fullDesign,1);
prefs.fullDesign = prefs.fullDesign([randperm(prefs.nTrials)],:);

% trial numbers for each of the conditions
for i = 1:prefs.nCond;
    prefs.cond(i).trialNums = find(prefs.fullDesign(:,end)==i);
end

% response keys
prefs.keys = [KbName('left') KbName('right') KbName('esc')];

if (savedata)
    % Data files
    prefs.fileName = [prefs.expName '_' prefs.dateStr];
    % prefs.fidxls = fopen(fullfile('output_xls',[prefs.fileName '.xls']), 'a');
    prefs.fidmat = fullfile('output_mat',[prefs.fileName '.mat']);
end
% ========================================================================
% CALCULATIONS BASED ON PREFERENCES (change not necessary)
% ========================================================================

% screen info (visual)
screenNumber = max(Screen('Screens'));       % use external screen if exists
[w, h]=Screen('WindowSize', screenNumber);  % screen resolution
screenResolution = [w h];       % screen resolution
screenCenter = screenResolution/2;       % screen center
screenDistance = 45;                      % distance between observer and screen (in cm)
screenAngle = atand((prefs.screenHeight/2) / screenDistance); % total visual angle of screen
screen_ppd = screenResolution(1) / screenAngle;  % pixels per degree
prefs.ellipseArea = prefs.ellipseArea * screen_ppd^2; % ellipse area (pixels)

% open screen
if ~(sessionnum)
    windowPtr = Screen('OpenWindow',screenNumber,prefs.grey,[],32,2);
else
    windowPtr = 10;
end
Screen(windowPtr,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
HideCursor;

% calculating where breaks will occur
if prefs.blocknum > 1;
    breakpoints = round((1:(prefs.blocknum-1)).* (prefs.nTrials/prefs.blocknum));
else
    breakpoints = prefs.nTrials+1;
end

if isempty(prefs.feedbacktrial); prefs.feedbacktrial = prefs.nTrials+1; end
% -----------------------------------------------------------------------
% matrix of all possible ellipse stimuli
% -----------------------------------------------------------------------
% (res degrees between each possible stimulus orientations)

clear StimPatches;

% Set the number of possible orientations based on resolution
res = 1; % resolution
numDegrees = ceil(180/res);

% StimPatches holds the ellipse image patches
StimPatches = zeros(length(prefs.reliabilityNum),numDegrees);

% Fill StimPatches
StimSizes = zeros(length(prefs.reliabilityNum),numDegrees,2);
for i=1:length(prefs.reliabilityNum)
    % Eccentricity = reliability for now
    ecc = prefs.reliabilityNum(i);
    % Draw a patch for each orientation
    for j = 1:numDegrees
        im = drawEllipse(prefs.ellipseArea, ecc, j*res,prefs.stimColor,prefs.bgColor);
        StimPatches(i,j) = Screen('MakeTexture',windowPtr,im);
        StimSizes(i,j,:) = size(im);
        StimSizes(i,j,:) = StimSizes(i,j,[2 1]);
    end
end

% line stimuli
prefs.lineArea = prefs.ellipseArea; % so line is same area as ellipse
prefs.lineWidth = 3;%round(prefs.lineArea*.01); % pixels
prefs.lineLength = round(prefs.lineArea/prefs.lineWidth);
lineCoordinates = nan(2,numDegrees);
for j = 1:numDegrees;
    [lineCoordinates(1,j), lineCoordinates(2,j)] = lineCoord(prefs.lineLength,j);
end
% prefs.lineRect = prefs.stimColor.*ones(prefs.lineWidth,prefs.lineLength);
% prefs.lineTexture = Screen('MakeTexture',windowPtr,prefs.lineRect);


% ========================================================================
% EXPERIMENT TRIAL AND STIMULI INFORMATION/CALCULATIONS
% ========================================================================
maxSetSize = max(prefs.setSizeNum);
etaScale = 0.15;

% TRIAL VARIABLES
% -------------------------------------------------------------------------

% DESIGN MAT: conditions and responses of experiment
designMat = nan(prefs.nTrials,8); % matrix of all conditions to use during trials
designMat(:,2) = prefs.setSizeNum(prefs.fullDesign(:,2));
designMat(:,3) = prefs.reliabilityNum(prefs.fullDesign(:,3));
designMat(:,4) = prefs.ISIdelayNum(prefs.fullDesign(:,4));
designMat(:,5) = prefs.fullDesign(:,5);
% = [ D.deltaNum D.setSizeNum D.reliabilityNum D.ISIdelayNum D.conditionNum D.response D.corrRespNum D.RT];
names.designMat{1} = 'delta'; names.designMat{2} = 'set size';
names.designMat{3} = 'ellipse reliability'; names.designMat{4} = 'delay time';
names.designMat{5} = 'condition number'; names.designMat{6} = 'response';
names.designMat{7} = 'Correct?'; names.designMat{8} = 'RT';

% STIMULI MAT: screen positions, orientations, target locations
stimuliMat = nan(prefs.nTrials, 4*maxSetSize + 1);
stimuliMat(:,end) = ceil(rand(prefs.nTrials,1).*designMat(:,2));
% names of columns
for j = 1:maxSetSize;
    names.stimuliMat{j} = ['stimulus ' num2str(j) ' x position'];
    names.stimuliMat{j+maxSetSize} = ['stimulus ' num2str(j) ' y position'];
    names.stimuliMat{j+2*maxSetSize} = ['stimulus' num2str(j) 'orientation (pres 1)'];
    names.stimuliMat{j+3*maxSetSize} = ['stimulus' num2str(j) 'orientation (pres 2)'];
end
names.stimuliMat{2*maxSetSize+1} = 'target location number';

% filling in pres1Orientations, XPositions, and YPositions.
for i = 1:prefs.nTrials
    % Initial orientations
    isetsize = designMat(i,2);
    randOrt = 180*rand(1,isetsize);
    stimuliMat(i,2*maxSetSize+1:2*maxSetSize+isetsize) = randOrt;
    
    % Pick angle of first one rand and add the rest in counterclockwise
    % fashion with angle spacing = 2pi/max(E.setSizeNum)
    locAngles = rand()*2*pi+(1:(maxSetSize))*(2*pi)/maxSetSize;
    [X, Y] = pol2cart(locAngles,screen_ppd * prefs.stimecc);
    % positions with jitter
    stimuliMat(i,1:isetsize) = X(1:isetsize) + screenCenter(1)...
        + round((rand(1,isetsize)-.5)*prefs.jitter*screen_ppd);
    stimuliMat(i,maxSetSize+1:maxSetSize+isetsize) = Y(1:isetsize) + screenCenter(2)...
        + round((rand(1,isetsize)-.5)*prefs.jitter*screen_ppd);
end

% ========================================================================
% RUN THE EXPERIMENT
% ========================================================================

% start/instruction screen
Screen('TextSize',windowPtr,28);
textx = 100;
texty = screenCenter(2) - 150;
dy = 30;
Screen('DrawText',windowPtr,'The orientation changed from the first and second stimuli.',textx,texty,[255 255 255]); texty = texty + 3*dy;

Screen('DrawText',windowPtr,'Press the LEFT ARROW if you think it rotated counterclockwise',textx,texty,[255 255 255]); texty = texty + dy;
Screen('DrawText',windowPtr,'Press the RIGHT ARROW if you think it rotated clockwise.',textx,texty,[255 255 255]); texty = texty + 3*dy;

[newx, newy] = Screen('DrawText',windowPtr,'These all have the same orientation:',textx,texty,[255 255 255]); texty = texty + 3*dy;

amntchange = 70;
xCoord = newx + amntchange;
for i = 1:length(prefs.reliabilityNum);
    cuesrcrect = [0 0 squeeze(StimSizes(i,round(numDegrees/4),:))'];
    destrect = CenterRectOnPoint(cuesrcrect,xCoord,newy+20);
    Screen('drawtexture',windowPtr,StimPatches(i,round(numDegrees/4)),cuesrcrect,destrect,0);
    xCoord = xCoord + amntchange;
end
xy = [lineCoordinates(:,round(numDegrees/4)), -lineCoordinates(:,round(numDegrees/4))];
Screen('DrawLines',windowPtr,xy,prefs.lineWidth,prefs.stimColor,[destrect(3)+amntchange destrect(4)-20],1);

Screen('Flip', windowPtr);
waitForKey;

% run a trial
% -------------------------------------------------------------------------

% step correct for each condition
stepCounter = zeros(1,prefs.nCond);

% first delta of each condition ~ U[15,25]
conddir(1,:) = prefs.directionNum(prefs.design(:,1));
if (prefs.vmprior)
    designMat(:,1) = circ_vmrnd(0,prefs.vmprior,[prefs.nTrials,1]); % delta prior ~ N(true,90/sqrt(12))
else
    designMat(:,1) = -45+rand(prefs.nTrials,1)*90; % delta prior ~ Unif[-45, 45]
end

for i = 1:prefs.nTrials;
    
    % setting values for current trial
    condition = designMat(i,5); % current condition
    delta = designMat(i,1); % amnt of change to target item
    pres1orientations = stimuliMat(i,2*maxSetSize+1:3*maxSetSize);
    pres2orientations = pres1orientations;
    setsize = designMat(i,2);
    xpositions = stimuliMat(i,1:maxSetSize);
    ypositions = stimuliMat(i,maxSetSize+1:2*maxSetSize);

    
    targetlocation = ceil(rand*setsize); % randomly drawing which stimulus is the target
    pres2orientations(targetlocation) = pres1orientations(targetlocation) + delta; % setting orientation of target stim
    
    % saving variables into a struct (for data analysis)
    stimuliMat(i,end) = targetlocation;
    stimuliMat(i,3*maxSetSize+1:4*maxSetSize) = pres2orientations;
    
    % adjusting number to be between 1-180 for stimulus presentations
    pres1orientations = mod(round(pres1orientations),180);
    pres1orientations(pres1orientations == 0) = 180;
    pres2orientations = mod(round(pres2orientations),180);
    pres2orientations(pres2orientations == 0) = 180;
    
    stimLine = lineCoordinates(:,pres2orientations);
    
    % initial fixation screen
    Screen('fillRect',windowPtr,prefs.bgColor);
    drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
    t0 = GetSecs();
    Screen('flip',windowPtr); %tic;
    while (GetSecs()-t0)<prefs.fix1Dur;
        % do nothing
    end
    
    
    % first stimulus presentation
    if (prefs.permLocInPres1) % if items are presented in permuted order
        k = randperm(setsize);
    else
        k = 1:setsize;
    end
    Screen('fillRect',windowPtr,prefs.bgColor);
    for j= 1:setsize
        srcrect = [0 0 squeeze(StimSizes(prefs.reliabilityNum == designMat(i,3),pres1orientations(k(j)),:))'];
        destrect = CenterRectOnPoint(srcrect,xpositions(k(j)),ypositions(k(j)));
        Screen('drawtexture',windowPtr,StimPatches(prefs.reliabilityNum == designMat(i,3),pres1orientations(j)),srcrect,destrect,0);
        if (prefs.simultPres1) % simultaneous presentation
            if (prefs.stimecc)
                drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            end
        else % sequential presentation
            if (prefs.stimecc)
                drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            end
            Screen('flip',windowPtr);
            t0 = GetSecs();
            while (GetSecs()-t0)<prefs.pres1Dur;
                % do nothing
            end
            %blank screen in between
            Screen('fillRect',windowPtr,prefs.bgColor);
            drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            Screen('flip',windowPtr);
            t0 = GetSecs();
            while (GetSecs()-t0)<prefs.pres1ISIDur;
                % do nothing
            end
        end
    end
    
    if (prefs.simultPres1)
        Screen('flip',windowPtr); 
        % tic
        t0 = GetSecs();
        while (GetSecs()-t0)<prefs.pres1Dur;
            % do nothing
        end
    end
    
    
    if (prefs.screenshot)
        grabSize = 2.5 * screen_ppd * prefs.stimecc;
        grabrect = CenterRectOnPoint([0 0 grabSize grabSize],screenCenter(1),screenCenter(2));
        im = Screen('getimage',windowPtr,grabrect);
        imwrite(im,['screenshots/stim_' num2str(i) '_X.png'],'png');
    end
    
    % blank screen (inter-stimulus interval)
    Screen('fillRect',windowPtr,prefs.bgColor);
    drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
    Screen('flip',windowPtr); 
    % fprintf('stimulus1 pres: %f \n', toc)
    % tic;
    t0 = GetSecs();
    while (GetSecs()-t0) < designMat(i,4);
        % do nothing
    end
    
    % second stimulus presentation
    Screen('fillRect',windowPtr,prefs.bgColor);
    if (prefs.stimecc)
        drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
    end
    if (prefs.allStimInPres2)
        
        for j= 1:setsize
            xy = [-stimLine stimLine]; %aspen, need to change this is setsize >1
            Screen('DrawLines',windowPtr, xy, prefs.lineWidth,prefs.stimColor,[xpositions(j) ypositions(j)],1);
        end
    else
        xy = [-stimLine stimLine];
        Screen('DrawLines',windowPtr, xy, prefs.lineWidth,prefs.stimColor,[xpositions(targetlocation) ypositions(targetlocation)],1);
    end
    Screen('flip',windowPtr); %tic;
    % fprintf('ISI: %f \n', toc)
    t0 = GetSecs();
    
    if (prefs.respInPres2) % if response is in 2nd stim presentation
        
    else % if response is not during 2nd presentation
        while (GetSecs()-t0)<prefs.pres2Dur;
            % do nothing
        end
        % blank screen (waiting for response)
        Screen('fillRect',windowPtr,prefs.bgColor);
        drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
        %             pres2time = toc
        Screen('flip',windowPtr);
    end
    
    
    % checking for response
    [pressedKey, designMat(i,8)] = waitForKeys(prefs.keys,GetSecs());
    if pressedKey == 1;
        response = -1;
    elseif pressedKey == 2;
        response = 1;
    elseif pressedKey == 3;
        sca;
        ShowCursor;
        fclose('all');
        clear all;
    end
    
    % correct/incorrect calculation
    if delta/abs(delta) == response;
        correct = 1;
    else
        correct = 0;
    end
    
    
    if (prefs.screenshot) % if you want a screenshot
        grabSize = 2.5 * screen_ppd * prefs.stimecc;
        grabrect = CenterRectOnPoint([0 0 grabSize grabSize],screenCenter(1),screenCenter(2));
        im = Screen('getimage',windowPtr,grabrect);
        imwrite(im,['screenshots/stim_' num2str(i) '_Y.png'],'png');
    end
    
    % blank space/colored feedback fixation (intertrial display)
    Screen('fillRect',windowPtr,prefs.bgColor);
    
    if (prefs.feedback)
        if (correct)
            beep = MakeBeep(1200,.2);
%             drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.bgColor,prefs.fixLength);
        else
            beep = MakeBeep(500,.2);
%             drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.bgColor,prefs.fixLength);
        end
    else
        drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.bgColor,prefs.fixLength);
        Screen('flip',windowPtr);
        t0 = GetSecs();
        while (GetSecs()-t0)<prefs.ITIDur;
            %         do nothing
        end
    end
    Snd('Open');
    Snd('Play',beep);
%     
    
    % assign deltaNum based on prev trials
    %     if find(prefs.cond(D.conditionNum(i)).trialNums == i) < prefs.nTrialsPerCond;
    %         idxx = prefs.cond(D.conditionNum(i)).trialNums(find(prefs.cond(D.conditionNum(i)).trialNums == i)+1);
    
    if ~(prefs.vmprior)
        if (mod(condition,2))
            cond2 = condition+1;
        else
            cond2 = condition-1;
        end
        idx1 = find(prefs.cond(condition).trialNums==i);
        idx2 = sum(prefs.cond(cond2).trialNums < i);
        %     display(idx1+idx2);
        if idx1>=20 && idx2>= 20;
            deltaMean1 = mean(designMat(prefs.cond(condition).trialNums(idx1-10:idx1),1));
            deltaMean2 =  mean(designMat(prefs.cond(cond2).trialNums(idx2-10:idx2),1));
            eta = abs(deltaMean1-deltaMean2)*etaScale;
        else
            eta = 3;
        end
    end
    
    % putting values back into designMat
    designMat(i,6) = response;
    designMat(i,7) = correct;

    if (savedata)
        save(prefs.fidmat,'designMat','stimuliMat','names')
    end
    
    % code for breaks/blocks
    if ~isempty(intersect(breakpoints,i)) || ~mod(i,prefs.feedbacktrial)
        Screen('fillRect',windowPtr,prefs.bgColor);
        t0 = GetSecs();
        Screen('flip',windowPtr);
        while GetSecs < .5;
            % do nothing
        end
        
        if (intersect(breakpoints,i) == breakpoints(ceil(prefs.blocknum/2)))
            
            Screen('fillRect',windowPtr,prefs.bgColor);
            Screen('DrawText',windowPtr,'You are halfway through this block. You are doing a great job! Press any key to continue',250,screenCenter(2) - 50,[255 255 255]);
            Screen('Flip', windowPtr);
            waitForKey;
            
            drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            Screen('Flip', windowPtr);
            WaitSecs(prefs.fix1Dur);
            
        elseif ~isempty(intersect(breakpoints,i))
            Screen('fillRect',windowPtr,prefs.bgColor);
            tic
            
            while toc<prefs.breakDuration
                Screen('DrawText',windowPtr,['You have finished ' num2str(find(breakpoints == i)) '/' num2str(length(breakpoints)+1) ' of the trials'],100,screenCenter(2)-80,[255 255 255]);
                Screen('DrawText',windowPtr,['Please take a short break now. You can continue in ' num2str(round(prefs.breakDuration-toc)) ' seconds…'],100,screenCenter(2)-100,[255 255 255]);
                Screen('Flip', windowPtr);
            end
            Screen('DrawText',windowPtr,'Press any key to continue',100,screenCenter(2)-80,prefs.white);
            Screen('Flip', windowPtr);
            waitForKey;
            
            drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            Screen('Flip', windowPtr);
            WaitSecs(prefs.fix1Dur);
            
        else
            progPC = mean(designMat(1:i,7));
            
            Screen('fillRect',windowPtr,prefs.bgColor);
            Screen('DrawText',windowPtr,['You got ' num2str(progPC*100) '% correct so far. Press any key to continue'],250,screenCenter(2) - 50,[255 255 255]);
            Screen('Flip', windowPtr);
            waitForKey;
            
            drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            Screen('Flip', windowPtr);
            WaitSecs(prefs.fix1Dur);
        end
        
    end
    
end

PC = mean(designMat(:,7)); % percent correct


% final screen
Screen('fillRect',windowPtr,prefs.bgColor);
Screen('DrawText',windowPtr,'End of this block. Press any key to continue',250,screenCenter(2) - 50,prefs.white);
Screen('Flip', windowPtr);
waitForKey;


% ==============
% GRAPHS AND PARAMETER FITTING
% ================
% addpath('C:/Users/Research/OneDrive/Research/VSTM/orientation_exp/analyses');


% proportion respond clockwise by delta (by condition)
% responsee = designMat(:,6);
% responsee(responsee == -1) = 0;
% responsee = [designMat(:,1) responsee];
% for i = 1:prefs.nCond/2;
%     data(i).Mat = sortrows([responsee(prefs.cond(2*i-1).trialNums,:);...
%         responsee(prefs.cond(2*i).trialNums,:)]);
% end
% 
% param_fit = nan(prefs.nCond/2,3);
% for i = 1:prefs.nCond/2;
%     param_fit(i,:) = fminsearch(@(x) nllobj(x(1),x(2),x(3),data(i).Mat(:,1)',data(i).Mat(:,2)'),[0,log(10),-log(1/.001-1)]);
% end
% % changing parameters back (exponentiating)
% param_fit(:,2) = exp(param_fit(:,2));
% param_fit(:,3) = exp(param_fit(:,3))./(exp(param_fit(:,3))+1);



% =======================================================================
% SAVE & CLOSE EVERYTHING
% ========================================================================
% writeTrialToFile(D, i, prefs.fidxls);
if (savedata)
    save(prefs.fidmat,'designMat','stimuliMat','names')
end

if ~(sessionnum);
ShowCursor;
sca;
end
% clear all;


% ========================================================================
% ---------------------------HELPER FUNCTIONS-----------------------------
% ========================================================================

function [pressedKey, RT] = waitForKeys(keys, tstart)

pressedKey=0;
while (1)
    
    [~, ~, keyCode] = KbCheck();
    if  any(keyCode(keys))
        RT = GetSecs - tstart;
        pressedKey = find(keyCode(keys));
        break;
    end
    
end

%----------------------------------------------------
function keyCode = waitForKey
keyCode = ones(1,256);
while sum(keyCode(1:254))>0
    [~,~,keyCode] = KbCheck;
end
while sum(keyCode(1:254))==0
    [~,~,keyCode] = KbCheck;
end
keyCode = min(keyCode==1);

%----------------------------------------------------
function drawfixation(windowPtr,x,y,color,size)
Screen('DrawLine',windowPtr,color,x-size,y,x+size,y,2);
Screen('DrawLine',windowPtr,color,x,y-size,x,y+size,2);
