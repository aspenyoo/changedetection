function runsession3(subjid,sessionnum) 
% 

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
    train_detect(subjid,sessionnum,40); % 400 trials of training
else
    train_detect(subjid,sessionnum,2); % 20 trials of training
end
Screen('Flip', windowPtr);

% ========== DETECTION TASK ============
Snd('Open');
for i = 1:length(beepVec);
    beep = MakeBeep(beepVec(i),0.1);
    Snd('Play',beep);
end
switch sessionnum
    case 1; nTrialsPerCond = 40;
    case 2; nTrialsPerCond = 87;
    case 3; nTrialsPerCond = 87;
    case 4; nTrialsPerCond = 86;
end

Exp_OrientationChangeDetect(subjid, sessionnum, nTrialsPerCond)
Screen('Flip', windowPtr);

% ========== THANK YOU SCREEN ==========
textx = 400;
texty = h/2 - 150;
Screen('DrawText',windowPtr,'Thanks for participating!',textx,texty,[255 255 255]);

Screen('Flip', windowPtr);
pause;

sca;