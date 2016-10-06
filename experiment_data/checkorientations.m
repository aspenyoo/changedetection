function checkorientations

commandwindow; 

Screen('Preference', 'ConserveVRAM', 64);
Screen('Preference', 'SkipSyncTests', 1);
screenNumber = max(Screen('Screens'));       % use external screen if exists
[w, h] = Screen('WindowSize', screenNumber);  % screen resolution of smaller display
windowPtr = Screen('OpenWindow',screenNumber,128*ones(1,3),[],32,2);
Screen(windowPtr,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

prefs = prefscode('Discrim','Simult','chk');

% screen info (visual)
screenResolution = [w h];       % screen resolution
screenDistance = 45;                      % distance between observer and screen (in cm)
screenAngle = 2*(180/pi)*(atan((prefs.screenWidth/2) / screenDistance)) ; % total visual angle of screen
screen_ppd = screenResolution(1) / screenAngle;  % pixels per degree
prefs.ellipseArea = prefs.ellipseArea * screen_ppd^2; % ellipse area (pixels)

prefs.lineArea = prefs.ellipseArea; % so line is same area as ellipse
prefs.lineWidth = 3;%round(prefs.lineArea*.01); % pixels
prefs.lineLength = round(prefs.lineArea/prefs.lineWidth);

stimx = w/2;
stimy = h/2;
for irelib = 1:length(prefs.reliabilityNum);
    for ideg = 1:180
        
        
        % circle
        ecc = prefs.reliabilityNum(irelib);
        im = drawEllipse(prefs.ellipseArea, ecc, ideg, prefs.stimColor, prefs.bgColor);
        StimPatch = Screen('MakeTexture',windowPtr,im);
        StimSize = size(im);
        cuesrcrect = [0 0 fliplr(StimSize)];
        destrect = CenterRectOnPoint(cuesrcrect,stimx,stimy);
        Screen('DrawTexture',windowPtr,StimPatch,cuesrcrect,destrect,0);
        
        
%         % line
%         [x,y] = lineCoord(prefs.lineLength, ideg);
%         Screen('DrawLine',windowPtr, prefs.stimColor-20, stimx+x, stimy+y, stimx-x, stimy-y, prefs.lineWidth);
        
        % flip
        Screen('Flip', windowPtr);
        pause;
        
    end
    
end