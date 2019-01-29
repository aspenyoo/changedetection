function run_experiment()

% Mixed reliability experiment

try
    
    %-%-%-%-%-
    %- INIT %-
    %-%-%-%-%-
    
    clear;
    
    % Get the experimental settings
    settings = getExperimentSettings();
    setsizeval = settings.setsizeval;
    reliabilityval = settings.reliabilityval;
    deltavartyperange = settings.deltavartyperange*(pi/180);
    reliabilitytype = settings.reliabilitytype;
	nTrials = settings.nTrials;
    breaknum = settings.breaknum;
    subjid = settings.subjid;
    stim_on_time = settings.stim_on_time;
    
    % set some condition-independent variables
    settings.makeScreenShot  = 0;    % if 1, then Screenshots of stimuli will be made
    settings.Screen_width    = 40;   % in cm (Dell@T115A: ~48cm; Dell@T101C: ~40 cm)
    settings.barwidth        = .3;   % width of stimulus bar (deg)
    settings.barheight       = .8;   % height of stimulus bar (deg)
    settings.ellipseArea     = settings.barwidth*settings.barheight; % ellipse size (deg^2)
    settings.jitter          = .6;   % amount of x/y-jitter (deg)
    settings.bgdac           = 128;  % background grayvalue (RGB)
    settings.fgdac           = 200;  % foreground grayvalue (RGB)
    settings.stimecc         = 7;    % stimulus eccentricity (deg)
    settings.ITT             = 1000;  % inter stimulus time (ms)
    settings.feedback        = 1;     % Set feedback flag    
    
    % Screen info
    ScreenNumber=max(Screen('Screens'));       % use external Screen if exists
    [w h]=Screen('WindowSize', ScreenNumber);  % Screen resolution
    Screen_resolution = [w h];                 % Screen resolution
    Screen_center = Screen_resolution/2;       % Screen center
    Screen_distance = 60;                      % distance between observer and Screen (in cm)
    Screen_angle = 2*(180/pi)*(atan((settings.Screen_width/2) / Screen_distance)) ; % total visual angle of Screen
    Screen_ppd = Screen_resolution(1) / Screen_angle;  % pixels per degree
    Screen_fixposxy = Screen_resolution .* [.5 .5]; % fixation position
    settings.ellipseArea = .3;
    settings.ellipseArea = settings.ellipseArea * Screen_ppd^2;
    
    % open Screen
    gray=GrayIndex(ScreenNumber);
    Screen('Preference', 'SkipSyncTests', 1);
    windowPtr = Screen('OpenWindow',ScreenNumber,gray,[],32,2);
    
    % Set break points
    breakpoints = round((1:(breaknum-1)).* (settings.nTrials/breaknum));
    
    % Fixation information
    fixsize = 4;
    fixcol = 0;
    nextFlipTime = 0; % just to initialize...
    aborted = 0;
    
    % Create trials
    % 1-> target presence (1,0)
    % 2-> response (1,0)
    % 3-> real stim time (ms?)
    % 4-> reaction time (ms)
    % 5-> number of stimuli [setsize] (int)
    % 6-> delta
    % 7:14-> first array orientations (radians)
    % 15:22-> second array orientations (radians)
    % 23:30-> x locations (pixels)
    % 31:38-> y locations (pixels)
    % 39:55-> eccentricity of ellipses
    
    % Draw stimuli randomly using generative model
    TrialMat = zeros(nTrials,55);
    
    % Set target presence/absence
    TrialMat(:,1) = rand(nTrials,1)>.5;
    
    % Set setsize
    TrialMat(:,5) = setsizeval(ceil(rand(nTrials,1)*length(setsizeval)));
    
    % Set deltas
    TrialMat(:,6) = (deltavartyperange*rand(nTrials,1));
    
    % Set reliabilities
    if (strcmp(reliabilitytype,'constant'))
        rel = reliabilityval(ceil(rand(nTrials,1)*length(reliabilityval)));
        size(rel)
        for i = 39:54
            TrialMat(:,i) = rel;
        end
    elseif (strcmp(reliabilitytype,'mixed'))
        rel = reliabilityval(ceil(rand(nTrials,8)*length(reliabilityval)));
        TrialMat(:,39:46) = rel;
        TrialMat(:,47:54) = rel;
    end
    
    % Set location of change
    TrialMat(:,1) = ceil(rand(nTrials,1).*TrialMat(:,1).*TrialMat(:,5));
    
    % Generate stimuli orientations, uniform over theta (random)
    for i = 1:nTrials
        % Initial orientations
        randOrt = (pi)*rand(1,TrialMat(i,5));
        TrialMat(i,7:(TrialMat(i,5)+7-1)) = randOrt;
        
        % Second orientations
        % Initially the same as the first
        TrialMat(i,15:(TrialMat(i,5)+15-1)) = randOrt;
        
        % Add change at target location
        TrialMat(i,14+TrialMat(i,1))=TrialMat(i,14+TrialMat(i,1))+TrialMat(i,6)*(TrialMat(i,1)~=0);
        
        % Pick angle of first one rand and add the rest in counterclockwise
        % fashion with angle spacing = 2pi/max(setsizeval)
        locAngles = rand()*2*pi+(1:(max(setsizeval)))*(2*pi)/max(setsizeval);
        [X,Y] = pol2cart(locAngles,Screen_ppd * settings.stimecc);
        TrialMat(i,23:(22+TrialMat(i,5))) = X(1:TrialMat(i,5)) + Screen_center(1);
        TrialMat(i,31:(30+TrialMat(i,5))) = Y(1:TrialMat(i,5)) + Screen_center(2);
        
    end   
    
    % Jitter stimulus positions
    TrialMat(:,23:38) = TrialMat(:,23:38) + round((rand(nTrials,16)-.5)*settings.jitter*Screen_ppd);
    
    % Create all stimulus patches, with degreeres 1 degree between each
    % possible stimulus orientations
    % Set the number of possible orientations based on resolution
    res = 1;
    numDegrees = ceil(180/res);
    
    % StimPatches holds the ellipse image patches
    clear StimPatches;
    StimPatches = zeros(length(reliabilityval),numDegrees);
    
    % Fill StimPatches
    StimSizes = zeros(length(reliabilityval),numDegrees+1,2);
    for i=1:length(reliabilityval)
       
        % Eccentricity = reliability for now
        ecc = reliabilityval(i);
        
        % Draw a patch for each orientation
        for j = 0:numDegrees
            b = sqrt(settings.ellipseArea * sqrt(1 - ecc^2) / pi);
            a = settings.ellipseArea / (pi*b);
            im = drawEllipse(2*b,2*a,j*res,settings.fgdac,settings.bgdac);
            StimPatches(i,j+1) = Screen('MakeTexture',windowPtr,im);
            StimSizes(i,j+1,:) = size(im);
            StimSizes(i,j+1,:) = StimSizes(i,j+1,[2 1]);            
        end
    end
    
    % Add on the actual degree values & StimPatch used for orientation
    % [exact_deg_i exact_deg_f StimPatch_idx_i StimPatch_dx_f]
    % Columns 56-63 and 64-71 contain the indices for getting images
    ExactDegs = [TrialMat(:,7:14)*(180/pi) TrialMat(:,15:22)*(180/pi)];
    ApproxDegs = [round(ExactDegs(:,1:8)*(1/res)) round(ExactDegs(:,9:16)*(1/res))];
    ApproxDegs = [mod(ApproxDegs(:,1:8),180)+1 mod(ApproxDegs(:,9:16),180)+1];
    TrialMat = [TrialMat ApproxDegs];
    
    % Randomize the trials
    TrialMat = TrialMat(randperm(nTrials),:);
    datafilename = ['output/' settings.subjid '/' settings.subjid '_' settings.expID '_' num2str(settings.nTrials) '_' datestr(now,'yyyymmddTHHMMSS') '.mat'];
    
    % Show start Screen. This needs to be done better
    Screen('TextSize',windowPtr,20);
    textx = 100;
    texty = Screen_center(2) - 50;
    dy = 32;
    cuesrcrect = [0 0 4+squeeze(StimSizes(1,round(numDegrees/4+1),:))'];
    Screen('DrawText',windowPtr,'Your task is to detect whether there is any change between',textx,texty,[255 255 255]); texty = texty + dy;
    Screen('DrawText',windowPtr,'the orientations of the first and second sets of ellipses',textx,texty,[255 255 255]); texty = texty + dy;
    [newx,newy] = Screen('DrawText',windowPtr,'The ORIENTATION of the ellipse looks like this:',textx,texty,[255 255 255]); texty = texty + dy;
    destrect = CenterRectOnPoint(cuesrcrect,newx+30,newy+20);
    Screen('drawtexture',windowPtr,StimPatches(length(reliabilityval),round(numDegrees/4+1)),cuesrcrect,destrect,0);
    Screen('DrawText',windowPtr,'The change can be of any magnitude. Press "y" if you think',textx,texty,[255 255 255]); texty = texty + dy;
    Screen('DrawText',windowPtr,'there is a change and "u" if you think they are the same.',textx,texty,[255 255 255]);
    
    
    Screen('Flip', windowPtr);
    waitForKey;
    Screen('fillRect',windowPtr,settings.bgdac);
    drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
    
    Screen('flip',windowPtr,nextFlipTime);
    nextFlipTime = GetSecs + 1;
    Screen('TextSize', windowPtr, 15);
    HideCursor;
    
    % Begin Trials!
    i = 0;
    while (i < nTrials) && ~aborted
        i = i+1;
        
        % SCREEN 1a: FIXATION
        Screen('fillRect',windowPtr,settings.bgdac);
        drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
        
        % First array, draw each element to screen
        for j= 1:TrialMat(i,5)
            squeeze(StimSizes(find(reliabilityval == TrialMat(i,38+j)),TrialMat(i,55+j),:));
            srcrect = [0 0 squeeze(StimSizes(find(reliabilityval == TrialMat(i,38+j)),TrialMat(i,55+j),:))'];
            destrect = CenterRectOnPoint(srcrect,TrialMat(i,22+j),TrialMat(i,30+j));
            Screen('drawtexture',windowPtr,StimPatches(find(reliabilityval == TrialMat(i,38+j)),TrialMat(i,55+j)),srcrect,destrect,0);
        end
        
        % show the stimuli for 0.1 seconds
        Screen('flip',windowPtr,nextFlipTime);
        
        nextFlipTime = GetSecs + stim_on_time;
        
        % SCREEN 1b: BLANK
        Screen('fillRect',windowPtr,settings.bgdac);
        drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
        Screen('flip',windowPtr,nextFlipTime);
        nextFlipTime = GetSecs + 1;
        
        % SCREEN 2: STIMULUS
        Screen('fillRect',windowPtr,settings.bgdac);
        drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
        
        % Second array, draw each element to screen
        for j= 1:TrialMat(i,5)
            srcrect = [0 0 squeeze(StimSizes(find(reliabilityval == TrialMat(i,38+8+j)),TrialMat(i,63+j),:))'];
            destrect = CenterRectOnPoint(srcrect,TrialMat(i,22+j),TrialMat(i,30+j));
            Screen('drawtexture',windowPtr,StimPatches(find(reliabilityval == TrialMat(i,38+8+j)),TrialMat(i,63+j)),srcrect,destrect,0);
        end
        
        % show the second stimulus for 0.1 seconds
        Screen('flip',windowPtr,nextFlipTime);
        stimStartTime = GetSecs;
        nextFlipTime = GetSecs + stim_on_time; 
        
        if (settings.makeScreenShot)
            grabSize = 2.5 * Screen_ppd * settings.stimecc;
            grabrect = CenterRectOnPoint([0 0 grabSize grabSize],Screen_fixposxy(1),Screen_fixposxy(2));
            im = Screen('getimage',windowPtr,grabrect);
            imwrite(im,['Screenshots/stim_' num2str(trialnr) '.png'],'png');
        end
        
        % SCREEN 3a: YES/NO RESPONSE
        Screen('fillRect',windowPtr,settings.bgdac);
        drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
        Screen('flip',windowPtr,nextFlipTime);
        REALSTIMTIME = round(1000*(GetSecs - stimStartTime));
        
        yesKey = KbName('y');
        noKey = KbName('u');
        escKey = KbName('esc');
        responseStartTime = GetSecs;
        done=0;
        while ~done
            keyCode = waitForKey;
            if (keyCode==yesKey)
                YESNORESP = 1;
                done=1;
            elseif (keyCode==noKey)
                YESNORESP = 0;
                done=1;
            elseif (keyCode == escKey)
                aborted=1;
                break;
            end
        end
        
        CORRECT = (TrialMat(i,1) && keyCode==yesKey) || (~TrialMat(i,1) && keyCode==noKey);
        RT = round(1000*(GetSecs - responseStartTime));
        
        TrialMat(i,2) = YESNORESP;
        TrialMat(i,3) = REALSTIMTIME;
        TrialMat(i,4) = RT;
        
        % save the data
        save(datafilename,'settings','TrialMat');
        
        % SCREEN 4: INTER TRIAL DISPLAY
        Screen('fillRect',windowPtr,settings.bgdac);
        if (settings.feedback)
            if (CORRECT)
                drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),[0 200 0],fixsize);
            else
                drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),[200 0 0],fixsize);
            end
        else
            drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
        end
        %Screen('DrawText',windowPtr,['Trial # = ' num2str(trialnr)],20,0,round(settings.bgdac*1.2));
        Screen('flip',windowPtr);
        nextFlipTime = GetSecs + (settings.ITT/1000);
        
        % show progress info + target reminder
        
        if ~isempty(intersect(breakpoints,i))
            Screen('fillRect',windowPtr,settings.bgdac);
            %drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
            %Screen('DrawText',windowPtr,['Trial # = ' num2str(trialnr)],20,0,round(settings.bgdac*1.2));
            Screen('flip',windowPtr,nextFlipTime);
            nextFlipTime = GetSecs + .5;
            
            if (intersect(breakpoints,i) == breakpoints(ceil(breaknum/2)))
                TempTrialMat = TrialMat(1:i,:);
                HalfPC = sum((TempTrialMat(:,1)>0)-TempTrialMat(:,2)==0)/i;
                
                Screen('fillRect',windowPtr,settings.bgdac);
                Screen('DrawText',windowPtr,['You are halfway. You got ' num2str(HalfPC*100) '% correct so far. Press <ENTER> to continue'],250,Screen_center(2) - 50,[255 255 255]);
                Screen('Flip', windowPtr);
                waitForKey;
                
                drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
                Screen('Flip', windowPtr);
                waitSecs(1.2);
                nextFlipTime = GetSecs + .5;
                
            else
                Screen('fillRect',windowPtr,settings.bgdac);
                tic
                cdtime = 60;   % time in seconds
                
                while toc<cdtime
                    Screen('DrawText',windowPtr,['You have finished ' num2str(round(100*i/settings.nTrials)) '% of the trials'],100,Screen_center(2)-80,[255 255 255]);
                    Screen('DrawText',windowPtr,['Please take a short break now. You can continue in ' num2str(round(cdtime-toc)) ' seconds…'],100,Screen_center(2)-100,[255 255 255]);
                    Screen('Flip', windowPtr);
                end
                Screen('DrawText',windowPtr,'Press any key to continue',100,Screen_center(2)-80,[255 255 255]);
                Screen('Flip', windowPtr);
                waitForKey;
                
                drawfixation(windowPtr,Screen_fixposxy(1),Screen_fixposxy(2),fixcol,fixsize);
                Screen('Flip', windowPtr);
                waitSecs(1.2);
                nextFlipTime = GetSecs + .5;
            end
            
        end
        
        % If this is a practice session, gradually shorten stimulus time
        if strcmp(settings.expID,'Practice') && (mod(i,32)==0)
            stim_on_time = stim_on_time - .033; 
        end
    end
    
    % Calculate percent correct
    PC = sum((TrialMat(1:i,1)>0)-TrialMat(1:i,2)==0)/i;
    
    % Compute 65% correct threshold        
    if strcmp(settings.expID,'Threshold')
        low_rel = compute_ellipse_thresholds(TrialMat,0);
        save(['./output/' subjid '/low_rel'],low_rel)
    end
    
    % show end screen
    Screen('fillRect',windowPtr,settings.bgdac);
    Screen('DrawText',windowPtr,['End of this session. You got ' num2str(PC*100) '% correct. Press <ENTER> to continue'],250,Screen_center(2) - 50,[255 255 255]);
    Screen('Flip', windowPtr);
    key = 0;
    while (key ~= 13)
        key = waitForKey;
    end
    
    % FINALIZE 

    ShowCursor;
    Screen('closeall');
    
catch
    % catch error
    Screen('CloseAll');
    ShowCursor;
    fclose('all');
    clear crash_variables;
    save crash_variables;
    psychrethrow(psychlasterror);
    
end % try ... catch %


%-%-%-%-%-%-%-%-%-%-%-%-%- HELPER FUNCTIONS %-%-%-%-%-%-%-%-%-%-%-%-%-%-%-

function keyCode = waitForKey
keyCode = ones(1,256);
while sum(keyCode(1:254))>0
    [keyIsDown,secs,keyCode] = KbCheck;
end
while sum(keyCode(1:254))==0
    [keyIsDown,secs,keyCode] = KbCheck;
end
keyCode = min(find(keyCode==1));

function drawfixation(windowPtr,x,y,color,size)
Screen('DrawLine',windowPtr,color,x-size,y,x+size,y,2);
Screen('DrawLine',windowPtr,color,x,y-size,x,y+size,2);


