function train_discrim(subjid,nTrialsPerCond)
if nargin < 2; nTrialsPerCond = 5; end

% training for orientation discrimination task
prefs = prefscode('Discrim','Simult',subjid);

% Screen('Preference', 'SkipSyncTests', 1);

% screen info (visual)
screenNumber = max(Screen('Screens'));       % use external screen if exists
[w, h] = Screen('WindowSize', screenNumber);  % screen resolution of smaller display
screenResolution = [w h];       % screen resolution
screenCenter = screenResolution/2;       % screen center
screenDistance = 45;                      % distance between observer and screen (in cm)
screenAngle = 2*(180/pi)*(atan((prefs.screenWidth/2) / screenDistance)) ; % total visual angle of screen
screen_ppd = screenResolution(1) / screenAngle;  % pixels per degree
prefs.ellipseArea = prefs.ellipseArea * screen_ppd^2; % ellipse area (pixels)

prefs.lineArea = prefs.ellipseArea; % so line is same area as ellipse
prefs.lineWidth = 3;%round(prefs.lineArea*.01); % pixels
prefs.lineLength = round(prefs.lineArea/prefs.lineWidth);

% open screen
windowPtr = 10;
% windowPtr = Screen('OpenWindow',screenNumber,prefs.grey,[],32,2);
Screen(windowPtr,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

Screen('TextSize',windowPtr,28);
Screen('TextFont',windowPtr,'Helvetica');
textx = 400;
texty = screenCenter(2) - 150;
dy = 30;


% ========== INTRODUCING STIMULI ==========
Screen('DrawText',windowPtr,'In this experiment, each trial consists of two stimuli.',textx,texty,[255 255 255]); texty = texty + 3*dy;
[newx, newy] = Screen('DrawText',windowPtr,'The first stimulus is an ellipse: ',textx,texty,[255 255 255]); texty = texty + 3*dy;

amntchange = 70;
newy = newy + 20;
xCoord = newx + 20;
degrot = 45;
for i = 1:length(prefs.reliabilityNum);
    ecc = prefs.reliabilityNum(i);
    im = drawEllipse(prefs.ellipseArea, ecc, degrot, prefs.stimColor, prefs.bgColor);
    StimPatch = Screen('MakeTexture',windowPtr,im);
    StimSize = size(im);
    cuesrcrect = [0 0 StimSize];
    destrect = CenterRectOnPoint(cuesrcrect,xCoord,newy);
    Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
    xCoord = xCoord + amntchange;
end

[newx, newy] = Screen('DrawText',windowPtr,'The second stimulus is a line:',textx,texty,[255 255 255]); texty = texty + dy;


[x,y] = lineCoord(prefs.lineLength, degrot);
newx = newx + 40; newy = newy + 20;
Screen('DrawLine',windowPtr, prefs.stimColor, newx+x, newy+y, newx-x, newy-y, prefs.lineWidth);

Screen('Flip', windowPtr);
pause;


% ========= SHAPE INVARIANCE ==========
texty = screenCenter(2) - 150;
[~, newy] = Screen('DrawText',windowPtr,'Note that all of these stimuli have the same orientation.',textx,texty,[255 255 255]); texty = texty + 3*dy;

amntchange = 100;
newy = newy + 150;
xCoord = screenCenter(1) - 200;
degrot = 45;
for i = 1:length(prefs.reliabilityNum);
    ecc = prefs.reliabilityNum(i);
    im = drawEllipse(prefs.ellipseArea, ecc, degrot, prefs.stimColor, prefs.bgColor);
    StimPatch = Screen('MakeTexture',windowPtr,im);
    StimSize = size(im);
    cuesrcrect = [0 0 fliplr(StimSize)];
    destrect = CenterRectOnPoint(cuesrcrect,xCoord,newy);
    Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
    xCoord = xCoord + amntchange;
end

newx = xCoord + 40; newy = newy + 10;
Screen('DrawLine',windowPtr, prefs.stimColor, newx+x, newy+y, newx-x, newy-y, prefs.lineWidth);

Screen('Flip', windowPtr);
pause;

% ========== EXPLANATION OF TASK ==========
texty = screenCenter(2) - 150;
Screen('DrawText',windowPtr,'An ellipse will flash up briefly, and the line shortly after',textx,texty,[255 255 255]); 
Screen('Flip', windowPtr);
pause;


% EXAMPLE TRIAL
% fixation
Screen('DrawText',windowPtr,'An ellipse will flash up briefly, and the line shortly after',textx,texty,[255 255 255]);
drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
Screen('Flip', windowPtr);
pause(0.5);

% stimulus 1
Screen('DrawText',windowPtr,'An ellipse will flash up briefly, and the line shortly after',textx,texty,[255 255 255]);
ecc = prefs.reliabilityNum(round(length(prefs.reliabilityNum)/2));
deg1 = round(rand*180);
im = drawEllipse(prefs.ellipseArea, ecc, deg1, prefs.stimColor, prefs.bgColor);
StimPatch = Screen('MakeTexture',windowPtr,im);
StimSize = size(im);
cuesrcrect = [0 0 fliplr(StimSize)];
destrect = CenterRectOnPoint(cuesrcrect,screenCenter(1),screenCenter(2));
Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
Screen('Flip', windowPtr);
pause(0.3);

% ISI
Screen('DrawText',windowPtr,'An ellipse will flash up briefly, and the line shortly after',textx,texty,[255 255 255]);
drawfixation(windowPtr,screenCenter(1),screenCenter(2),prefs.fixColor,prefs.fixLength);
Screen('Flip', windowPtr);
pause(1);

% stimulus 2
Screen('DrawText',windowPtr,'An ellipse will flash up briefly, and the line shortly after',textx,texty,[255 255 255]);
deg2 = deg1+15;
[x,y] = lineCoord(prefs.lineLength, deg2);
Screen('DrawLine',windowPtr, prefs.stimColor, screenCenter(1)+x, screenCenter(2)+y, screenCenter(1)-x, screenCenter(2)-y, prefs.lineWidth);
Screen('Flip', windowPtr);
pause;


% ========== GAUSSIAN PRIOR OVER CHANGE ==========
texty = screenCenter(2) - 150;
Screen('DrawText',windowPtr,'The orientation from the ellipse to line will ALWAYS CHANGE.',textx,texty,[255 255 255]); texty = texty + 3*dy;
Screen('DrawText',windowPtr,'Small changes are more probable than large changes',textx,texty,[255 255 255]); texty = texty + 5*dy;

% EXAMPLES
Screen('DrawText',windowPtr,'Examples:',textx,texty,[255 255 255]);
Screen('Flip', windowPtr);
pause;

nExamples = 4; nSamples = 20;
newx = textx;
for iexamp = 1:nExamples;
    texty = screenCenter(2) - 150;
    Screen('DrawText',windowPtr,'The orientation from the ellipse to line will ALWAYS CHANGE.',textx,texty,[255 255 255]); texty = texty + 3*dy;
    Screen('DrawText',windowPtr,'Small changes are more probable than large changes',textx,texty,[255 255 255]); texty = texty + 5*dy;

    % EXAMPLES
    Screen('DrawText',windowPtr,'Examples:',textx,texty,[255 255 255]);
    
    % draw one ellipse
    deg1 = round(rand*180);
    ecc = prefs.reliabilityNum(randi(length(prefs.reliabilityNum)));
    im = drawEllipse(prefs.ellipseArea, ecc, deg1, prefs.stimColor, prefs.bgColor);
    StimPatch = Screen('MakeTexture',windowPtr,im);
    StimSize = size(im);
    cuesrcrect = [0 0 fliplr(StimSize)];
    newx = newx + 200; newy = texty + 150;
    destrect = CenterRectOnPoint(cuesrcrect,newx,newy);
    Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
    Screen('Flip', windowPtr);
    pause;
   
    for isamp = 1:nSamples;
        
        % redrawing the same stuff that was already there
        texty = screenCenter(2) - 150;
        Screen('DrawText',windowPtr,'The orientation from the ellipse to line will ALWAYS CHANGE.',textx,texty,[255 255 255]); texty = texty + 3*dy;
        Screen('DrawText',windowPtr,'Small changes are more probable than large changes',textx,texty,[255 255 255]); texty = texty + 5*dy;
        Screen('DrawText',windowPtr,'Examples:',textx,texty,[255 255 255]);
        Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
        
        % draw a bunch of lines from that one ellipse
        deg2 = deg1+randn*prefs.gaussprior;
        [x,y] = lineCoord(prefs.lineLength, deg2);
        Screen('DrawLine',windowPtr, prefs.stimColor-20, newx+x, newy+y, newx-x, newy-y, prefs.lineWidth);
        
        Screen('Flip', windowPtr);
        pause(0.15);
    end
    pause;
end


% ========== RESPONSE BUTTONS ==========
texty = screenCenter(2) - 150;
Screen('DrawText',windowPtr,'Your task is to indicate the DIRECTION of change.',textx,texty,[255 255 255]); texty = texty + 5*dy;
Screen('DrawText',windowPtr,'Clockwise change: <RIGHT ARROW KEY>',textx,texty,[255 255 255]); texty = texty + 3*dy;
Screen('DrawText',windowPtr,'Counter-clockwise change: <LEFT ARROW KEY>',textx,texty,[255 255 255]);

Screen('Flip', windowPtr);
pause;

% ========== ANY QUESTIONS? ==========
textx = screenCenter(1) - 75;
texty = screenCenter(2) - 70;
Screen('DrawText',windowPtr,'Any questions?',textx,texty,[255 255 255]);
Screen('Flip', windowPtr);
pause;


% ========== RUNNING DISCRIM TASK FOR A LITTLE ==========
Screen('Flip', windowPtr);
Exp_OrientationChangeDiscrim(subjid, 99, nTrialsPerCond);

end

% =====================================================================
% HELPER FUNCTIONS 
% =====================================================================

function drawfixation(windowPtr,x,y,color,size)
Screen('DrawLine',windowPtr,color,x-size,y,x+size,y,2);
Screen('DrawLine',windowPtr,color,x,y-size,x,y+size,2);
end