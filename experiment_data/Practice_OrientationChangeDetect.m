function Practice_OrientationChangeDetect(subjid, sessionnum,nTrialsPerCond)
% runs 2AFC experiment: orientation change detection task. 
% 
% STRUCT EXPLANATIONS
% prefs: experimental preferences
% D: related to data (things you may analyze about the experiment in the
% future)

% ========================================================================
% SETTING THINGS UP (changing this section is usually unnecessary)
% ========================================================================
if nargin < 1; subjid = []; end
if nargin < 2; sessionnum = []; end
if nargin < 3; nTrialsPerCond = []; end

commandwindow;

% random number generator
rng('shuffle');

% ==================================================================
% PREFERENCES (should be changed to match experiment preferences)
% ==================================================================

% importing preferences for simultaneous/sequential experiment
prefs = prefscode('Pract_Detect','Simult',subjid, sessionnum,nTrialsPerCond);

% response keys
prefs.keys = [KbName('s') KbName('d') KbName('esc')];

prefs.keysNum = [1 0];

% Data file
prefs.fileName = [prefs.expName '_' prefs.dateStr];
% prefs.fidxls = fopen(fullfile('output_xls',[prefs.fileName '.xls']), 'a');
prefs.fidmat = fullfile('output_mat',[prefs.fileName '.mat']);

% calculating full experimental design and pseudo-randomizing order
prefs.design = fullfact([prefs.f1 prefs.f2 prefs.f3 prefs.f4]);
prefs.nCond = size(prefs.design,1);

% ========================================================================
% CALCULATIONS BASED ON PREFERENCES (change not necessary)
% ========================================================================
% skipping sync tests
% Screen('Preference', 'SkipSyncTests', 1);

% screen info (visual)
screenNumber =  max(Screen('Screens'));       % use external screen if exists
[w, h] = Screen('WindowSize', screenNumber);  % screen resolution of smaller display
screenResolution = [w h];                 % screen resolution
screenCenter = screenResolution/2;       % screen center
screenDistance = 45;                      % distance between observer and screen (in cm)
screenAngle = 2*(180/pi)*(atan((prefs.screenWidth/2) / screenDistance)) ; % total visual angle of screen
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

% ========================================================================
% EXPERIMENT TRIAL AND STIMULI INFORMATION/CALCULATIONS
% ========================================================================
maxSetSize = max(prefs.setSizeNum);


% TRIAL VARIABLES
% -------------------------------------------------------------------------
    
% DESIGN MAT: conditions and responses of experiment
designMat = prefs.design;
% designMat = nan(prefs.nTrials,8); % matrix of all conditions to use during trials
designMat(:,1) = prefs.deltaNum(prefs.design(:,1));
designMat(:,2) = prefs.setSizeNum(prefs.design(:,2));
designMat(:,3) = prefs.reliabilityNum(prefs.design(:,3));
designMat(:,4) = prefs.ISIdelayNum(prefs.design(:,4));
designMat(:,5) = 1:prefs.nCond;
if (prefs.vmprior)
    designMat = repmat(designMat, [prefs.nTrialsPerCond 1]);
    designMat(:,1) = designMat(:,1).*circ_vmrnd(0,prefs.vmprior,[size(designMat,1),1])*180/pi; %sort(repmat([-45:10:45]',[prefs.nCond 1]));
else
    designMat = repmat(designMat,[10,1]);
    designMat(:,1) = designMat(:,1).*sort(repmat([-45:10:45]',[prefs.nCond 1]));
    designMat = repmat(designMat, [prefs.nTrialsPerCond 1]);
end
designMat = [designMat nan(size(designMat,1),3)];
prefs.nTrials = size(designMat,1);
designMat = designMat([randperm(prefs.nTrials)],:);

% names of variables in designMat
names(1).designMat = 'delta'; names(2).designMat = 'set size';
names(3).designMat = 'ellipse reliability'; names(4).designMat = 'delay time';
names(5).designMat = 'condition number'; names(6).designMat = 'response';
names(7).designMat = 'Correct?'; names(8).designMat = 'RT';


% calculating where breaks will occur
if prefs.blocknum >1;
    prefs.breakpoints = round((1:(prefs.blocknum-1)).* (prefs.nTrials/prefs.blocknum));
else
    prefs.breakpoints = prefs.nTrials+1;
end

% D.deltaNum(:,1) = prefs.deltaNum([prefs.fullDesign(:,1)]);
% if ~(prefs.directionChange)
%     D.deltaNum = D.deltaNum.* round(180*rand(size(prefs.fullDesign,1),1)-90);
% end
% D.setSizeNum(:,1) = prefs.setSizeNum([prefs.fullDesign(:,2)]);
% D.reliabilityNum(:,1) = prefs.reliabilityNum([prefs.fullDesign(:,3)]);  %ellipse eccentricities
% D.ISIdelayNum(:,1) = prefs.ISIdelayNum([prefs.fullDesign(:,4)]);

% STIMULI MAT: screen positions, orientations, target locations
stimuliMat = nan(prefs.nTrials, 4*maxSetSize + 1);
% names of columns
for j = 1:maxSetSize;
names(j).stimuliMat = ['stimulus ' num2str(j) ' x position']; 
names(j+maxSetSize).stimuliMat = ['stimulus ' num2str(j) ' y position'];
names(j+2*maxSetSize).stimuliMat = ['stimulus' num2str(j) 'orientation (pres 1)'];
names(j+3*maxSetSize).stimuliMat = ['stimulus' num2str(j) 'orientation (pres 2)'];
end
names(4*maxSetSize+1).stimuliMat = 'target location number';

stimuliMat(:,end) = 1-(designMat(:,1)==0);
stimuliMat(:,end) = ceil(rand(prefs.nTrials,1).*stimuliMat(:,end).*designMat(:,2));

for i = 1:prefs.nTrials
    % Initial orientations
    isetsize = designMat(i,2);
    randOrt = round(180*rand(1,isetsize));
    stimuliMat(i,2*maxSetSize+1:2*maxSetSize+isetsize) = randOrt;
    
    % Second orientations
    stimuliMat(i,3*maxSetSize+1:3*maxSetSize+isetsize) = randOrt;
    % Add change at target location
    targetloc = stimuliMat(i,end);
    if (targetloc);
        stimuliMat(i,3*maxSetSize+targetloc)=stimuliMat(i,3*maxSetSize+targetloc)-designMat(i,1);
    end
%     D.pres2Orientations = mod(D.pres2Orientations,180);
%     D.pres2Orientations(D.pres2Orientations == 0) = 180;
    
    % Pick angle of first one rand and add the rest in counterclockwise
    % fashion with angle spacing = 2pi/max(E.setSizeNum)
    locAngles = rand()*2*pi+(1:(max(prefs.setSizeNum)))*(2*pi)/max(prefs.setSizeNum);
    [X, Y] = pol2cart(locAngles, screen_ppd * prefs.stimecc);
    % positions with jitter
    stimuliMat(i,1:isetsize) = X(1:isetsize) + screenCenter(1)...
        + round((rand(1,isetsize)-.5)*prefs.jitter*screen_ppd);
    stimuliMat(i,maxSetSize+1:maxSetSize+isetsize) = Y(1:isetsize) + screenCenter(2)...
        + round((rand(1,isetsize)-.5)*prefs.jitter*screen_ppd);
    
end

% matrix of all possible stimuli
% -----------------------------------------------------------------------
% (res degrees between each possible stimulus orientations)

clear StimPatches;

% Set the number of possible orientations based on resolution
res = 1; % resolution
numDegrees = ceil(180/res);

StimPatches = zeros(length(prefs.reliabilityNum),numDegrees); % holds the ellipse image patches
StimSizes = zeros(length(prefs.reliabilityNum),numDegrees+1,2); % holds ellipse image sizes

% Fill StimPatches and StimSizes
for i = 1:length(prefs.reliabilityNum)
    % Eccentricity = reliability for now
    ecc = prefs.reliabilityNum(i);
    % Draw a patch for each orientation
    for j = 1:numDegrees
        im = drawEllipse(prefs.ellipseArea,ecc,j*res,prefs.stimColor,prefs.bgColor);
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

% ========================================================================
% RUN THE EXPERIMENT
% ========================================================================
dy = 30;
textx = 270;

if (sessionnum)
% ====== DETECTION INFO SCREEN =======
texty = screenCenter(2) - 200;
Screen('TextSize',windowPtr,28);
Screen('TextFont',windowPtr,'Helvetica');
Screen('DrawText',windowPtr,'The orientation will be the same in HALF of the trials.',textx,texty,[255 255 255]); texty = texty + 5*dy;
Screen('DrawText',windowPtr,'Your task is to indicate whether orientations of the two stimuli are the same or different.',textx,texty,[255 255 255]); 
Screen('Flip', windowPtr);
waitForKey;
end

% ====== BUTTON DIRECTIONS ======
texty = screenCenter(2) - 200;
if (prefs.directionChange)
    dirtext1 = 'The rotation can be of any magnitude. Press "y" if you think it';
    dirtext2 = 'rotated counterclockwise and "u" if you think it rotated clockwise.';
else
    dirtext1 = 'Press "D" if the orientations of the shapes are different';
    dirtext2 = 'and "S" if the orientations of the shapes are the same.';
end

% Screen('DrawText',windowPtr,'Your task is to detect whether the ORIENTATION of the ',textx,texty,[255 255 255]); texty = texty + dy;
% Screen('DrawText',windowPtr,'first and second stimuli are the SAME (no change in  ',textx,texty,[255 255 255]); texty = texty + dy;
% Screen('DrawText',windowPtr,'orientation) or are DIFFERENT (change in orientation).',textx,texty,[255 255 255]); texty = texty + 3*dy;

Screen('DrawText',windowPtr,dirtext1,textx,texty,[255 255 255]); texty = texty + dy;
Screen('DrawText',windowPtr,dirtext2,textx,texty,[255 255 255]); texty = texty + 4*dy;

Screen('DrawText',windowPtr,'Remember, a change in shape does not mean a change in orientation.',textx,texty,[255 255 255]); texty = texty + 2*dy;

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

Screen('DrawText',windowPtr,'Press any key to begin!',textx,texty,[255 255 255]); texty = texty + 3*dy;
Screen('Flip', windowPtr);
waitForKey;

% run a trial
% -------------------------------------------------------------------------

for i = 1:prefs.nTrials;
    
    % setting values for current trial
    condition = designMat(i,5); % current condition
    pres1orientations = stimuliMat(i,2*maxSetSize+1:3*maxSetSize);
    pres2orientations = stimuliMat(i,3*maxSetSize+1:4*maxSetSize);
    setsize = designMat(i,2);
    xpositions = stimuliMat(i,1:maxSetSize);
    ypositions = stimuliMat(i,maxSetSize+1:2*maxSetSize);
    
    % adjusting number to be between 1-180 for stimulus presentations
    pres1orientations = mod(round(pres1orientations),180);
    pres1orientations(pres1orientations == 0) = 180;
    pres2orientations = mod(round(pres2orientations),180);
    pres2orientations(pres2orientations == 0) = 180;
    
    lineStim = lineCoordinates(:,pres2orientations);
    
    % initial fixation screen
    Screen('fillRect',windowPtr,prefs.bgColor);
    drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
    t0 = GetSecs();
    Screen('flip',windowPtr);
    while (GetSecs()-t0)<prefs.fix1Dur;
        % do nothing
    end

    % first stimulus presentation
    
    Screen('fillRect',windowPtr,prefs.bgColor);
    if (prefs.permLocInPres1)
        k = randperm(designMat(i,2));
    else
        k = 1:designMat(i,2);
    end
    for j= 1:designMat(i,2)
        srcrect = [0 0 squeeze(StimSizes(prefs.reliabilityNum == designMat(i,3),pres1orientations(k(j)),:))'];
        destrect = CenterRectOnPoint(srcrect,xpositions(k(j)),ypositions(k(j)));
        Screen('drawtexture',windowPtr,StimPatches(prefs.reliabilityNum == designMat(i,3),pres1orientations(k(j))),srcrect,destrect,0);
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
    % fprintf('stims1 presentation: %f \n', toc)
    % tic
    t0 = GetSecs();
    while (GetSecs()-t0)<designMat(i,4);
        % do nothing
    end
    
    % second stimulus presentation
    Screen('fillRect',windowPtr,prefs.bgColor);
    if (prefs.stimecc)
    drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
    end
    if (prefs.allStimInPres2)
        
        for j= 1:designMat(i,2)
%             srcrect = [0 0 squeeze(StimSizes(prefs.reliabilityNum == designMat(i,3),D.pres2Orientations(i,j),:))'];
%             destrect = CenterRectOnPoint(srcrect,D.XPositions(i,j),D.YPositions(i,j));
%             Screen('drawtexture',windowPtr,StimPatches(prefs.reliabilityNum == designMat(i,3),D.pres2Orientations(i,j)),srcrect,destrect,0);
       xy = [-lineStim lineStim ]; % aspen, need to change this is setsize > 1
                Screen('DrawLines',windowPtr, xy, prefs.lineWidth,prefs.stimColor,[xpositions(j) ypositions(j)],1);
          
        end
    else
        j = stimuliMat(i,end);
        if j == 0;
            j = ceil(rand*designMat(i,2));
        end
%         srcrect = [0 0 squeeze(StimSizes(prefs.reliabilityNum == designMat(i,3),D.pres2Orientations(i,j),:))'];
%         destrect = CenterRectOnPoint(srcrect,D.XPositions(i,j),D.YPositions(i,j));
%         Screen('drawtexture',windowPtr,StimPatches(prefs.reliabilityNum == designMat(i,3),D.pres2Orientations(i,j)),srcrect,destrect,0);
    xy = [-lineStim lineStim ];
            Screen('DrawLines',windowPtr, xy, prefs.lineWidth,prefs.stimColor,[xpositions(j) ypositions(j)],1);
       
    end
    Screen('flip',windowPtr);
    % fprintf('ISI: %f \n', toc)
    t0 = GetSecs();
    
    if (prefs.respInPres2) % if response is in 2nd stim presentation
        % leave pres2 stimuli on screen until response    
    else % if response is not during 2nd presentation
        % leave pres2 stimuli on screen for fixed duration
        while (GetSecs()-t0)<prefs.pres2Dur;
            % do nothing
        end
        % blank screen (waiting for response)
        Screen('fillRect',windowPtr,prefs.bgColor);
        drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
        Screen('flip',windowPtr);
    end
    
    % check response
        [pressedKey, designMat(i,8)] = waitForKeys(prefs.keys,GetSecs());
        if pressedKey == 1 || pressedKey == 2;
            designMat(i,6) = prefs.keysNum(pressedKey);
        elseif pressedKey == 3;
            sca;
            ShowCursor;
            fclose('all');
            clear all;
        end
        
    % correct/incorrect calculation
    if (prefs.directionChange)
        if (designMat(i,1)<0 && designMat(i,6)==prefs.keysNum(1)) || (designMat(i,1)>0 && designMat(i,6)==prefs.keysNum(2));
            designMat(i,7) = 1;
        else
            designMat(i,7) = 0;
        end
        
        if (prefs.screenshot) % if you want a screenshot
            grabSize = 2.5 * screen_ppd * prefs.stimecc;
            grabrect = CenterRectOnPoint([0 0 grabSize grabSize],screenCenter(1),screenCenter(2));
            im = Screen('getimage',windowPtr,grabrect);
            imwrite(im,['screenshots/stim_' num2str(i) '_Y.png'],'png');
        end
    else
        if (designMat(i,1)~=0 && designMat(i,6)==prefs.keysNum(2)) || (designMat(i,1)==0 && designMat(i,6)==prefs.keysNum(1));
            designMat(i,7) = 1;
        else
            designMat(i,7) = 0;
        end
    end
        
        
    % blank space/colored feedback fixation (intertrial display)
    Screen('fillRect',windowPtr,prefs.bgColor);
    
    if (prefs.feedback)
        if (designMat(i,7))
            beep = MakeBeep(1200,.2);
        else
            beep = MakeBeep(500,.2);
        end
        Snd('Open');
        Snd('Play',beep);
        %
    else
        drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.bgColor,prefs.fixLength);
        Screen('flip',windowPtr);
        t0 = GetSecs();
    end
    
    while (GetSecs()-t0)<prefs.ITIDur;
        % do nothing
    end
    
    % save trial in data file and mat file
%     if (i==1)
%         writeHeaderToFile(D, prefs.fidxls);
%     end
%     writeTrialToFile(D, i, prefs.fidxls);
    save(prefs.fidmat)
    
    % code for breaks/blocks
    if ~isempty(intersect(prefs.breakpoints,i))|| ~mod(i,prefs.feedbacktrial)
        Screen('fillRect',windowPtr,prefs.bgColor);
        t0 = GetSecs();
        Screen('flip',windowPtr);
        while GetSecs < .5;
            % do nothing
        end

%         if (intersect(prefs.breakpoints,i) == prefs.breakpoints(ceil(prefs.blocknum/2)))
% %             HalfPC = mean(designMat(1:i,7));
%             
%             Screen('fillRect',windowPtr,prefs.bgColor);
%             Screen('DrawText',windowPtr,['You are halfway through the second task. Great job! Press any key to continue'],250,screenCenter(2) - 50,[255 255 255]);
%             Screen('Flip', windowPtr);
%             waitForKey;
%             
%             drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
%             Screen('Flip', windowPtr);
%             WaitSecs(prefs.fix1Dur);
            
        if ~isempty(intersect(prefs.breakpoints,i)) %break
            Screen('fillRect',windowPtr,prefs.bgColor);
            tic
            
            while toc<prefs.breakDuration
                Screen('DrawText',windowPtr,['You have finished ' num2str(find(prefs.breakpoints == i)) '/' num2str(length(prefs.breakpoints)+1) ' of the trials.'],100,screenCenter(2)-80,[255 255 255]);
                Screen('DrawText',windowPtr,['Please take a short break now. You can continue in ' num2str(round(prefs.breakDuration-toc)) ' seconds.'],100,screenCenter(2)-120,[255 255 255]);
                Screen('Flip', windowPtr);
            end
            Screen('DrawText',windowPtr,'Press any key to continue',100,screenCenter(2)-80,prefs.white);
            Screen('Flip', windowPtr);
            waitForKey;
            
            drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
            Screen('Flip', windowPtr);
            WaitSecs(prefs.fix1Dur);
            
        else  % every FEEDBACKTRIALth trial
%             progPC = mean(designMat(1:i,7));
            
            Screen('fillRect',windowPtr,prefs.bgColor);
            Screen('DrawText',windowPtr,'You are doing great! Press any key to continue',250,screenCenter(2) - 50,[255 255 255]);
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
Screen('DrawText',windowPtr,'End of the practice. Press <ENTER> to continue',250,screenCenter(2) - 50,prefs.white);
Screen('Flip', windowPtr);
key = 0;
while (key ~= 13)
    key = waitForKey;
end

% =======================================================================
% SAVE & CLOSE EVERYTHING
% ========================================================================
% writeTrialToFile(D, i, prefs.fidxls);
save(prefs.fidmat)
    
if ~(sessionnum)
ShowCursor;
sca;
end


% ========================================================================
% ---------------------------HELPER FUNCTIONS-----------------------------
% ========================================================================

function [pressedKey, RT] = waitForKeys(keys, tstart)

pressedKey=0;
while (1)
    
    [keyIsDown, secs, keyCode] = KbCheck();
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
    [keyIsDown,secs,keyCode] = KbCheck;
end
while sum(keyCode(1:254))==0
    [keyIsDown,secs,keyCode] = KbCheck;
end
keyCode = min(find(keyCode==1));

%----------------------------------------------------
function drawfixation(windowPtr,x,y,color,size)
Screen('DrawLine',windowPtr,color,x-size,y,x+size,y,2);
Screen('DrawLine',windowPtr,color,x,y-size,x,y+size,2);


% ---------------------------------------------------
% function writeHeaderToFile(D, fid) 
%  
% h = fieldnames(D); 
%  
% for i=1:length(h) 
%     fprintf(fid, '%s\t', h {i }); 
% end
% fprintf(fid, '\n');
% 
% 
% % ------------------------------------------------------
% function writeTrialToFile(D, trial, fid) 
%  
% h = fieldnames(D); 
% for i=1:length(h) 
%     data = D.(h {i })(trial); 
%     if isnumeric(data)    
%         fprintf(fid, '%s\t', num2str(data)); 
%     elseif iscell(data) 
%         fprintf(fid, '%s\t', char(data)); 
%     else 
%         error('wrong format!') 
%     end 
% end      
% fprintf(fid, '\n'); 
