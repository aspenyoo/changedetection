function runsession(subjid,sessionnum) 
% runs the entire training + ABBBBBA discrim/detect design for one session

clc;

if nargin < 1; subjid = input('enter 3 letter subject ID: ', 's'); end
if nargin < 2; sessionnum = input('enter session number: '); end

beepVec = [1200 1000 800 1000 1200];

screenNumber = max(Screen('Screens'));       % use external screen if exists
[w, h] = Screen('WindowSize', screenNumber);  % screen resolution of smaller display
windowPtr = Screen('OpenWindow',screenNumber,128*ones(1,3),[],32,2);
Screen(windowPtr,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
Screen('TextSize',windowPtr,28);
Screen('TextFont',windowPtr,'Helvetica');
HideCursor;

% ========== TRAINING SESSION ON FIRST DAY ==========
if sessionnum == 1; 
    train_discrim(subjid);
    Snd('Open');
    for i = 1:length(beepVec);
        beep = MakeBeep(beepVec(i),0.1);
        Snd('Play',beep);
    end
end

Screen('Flip', windowPtr);

% ========== DISCRIM TASK ==========
Exp_OrientationChangeDiscrim(subjid, sessionnum)
Screen('Flip', windowPtr);

% ========== DETECTION TASK ============
for i = 1:length(beepVec);
    beep = MakeBeep(beepVec(i),0.1);
    Snd('Play',beep);
end

Exp_OrientationChangeDetect(subjid, sessionnum)
Screen('Flip', windowPtr);

% ========== DISCRIMINATION TASK ==========
for i = 1:length(beepVec);
    beep = MakeBeep(beepVec(i),0.1);
    Snd('Play',beep);
end

Exp_OrientationChangeDiscrim(subjid, sessionnum)
Screen('Flip', windowPtr);
pause(1)

% ========== THANK YOU SCREEN ==========
textx = 400;
texty = h/2 - 150;
Screen('DrawText',windowPtr,'Thanks for participating!',textx,texty,[255 255 255]);

Screen('Flip', windowPtr);
pause;

sca;