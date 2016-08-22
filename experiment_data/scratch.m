%% checking PC for subject
subjid = '1';
experimenttype = 'Detection';
sessionnumbers = {'1','2','3','4'};
isgaussprior = 1; 

[data] = concatdata(subjid, experimenttype, sessionnumbers);
Xdet = conditionSeparator(data,isgaussprior);

nCond = length(Xdet);
for icond = 1:nCond;
    dta = Xdet{icond};
    
    dta(:,1) = dta(:,1)==0;
    PC(icond) = mean(dta(:,1) == dta(:,2));
end
PC


% p(respond same) as a function of change (deg)
[stimLevelsCell, trialNumsCell,nRespCell] = conditionSeparator(data,isgaussprior);
blues = aspencolors(nCond,'blue');
figure(1); clf; hold on
for icond = 1:nCond;
    plot(stimLevelsCell{icond},nRespCell{icond}./trialNumsCell{icond},'Color',...
        blues(icond,:),'LineWidth',2);
end
defaultplot



%% plotting VM and Gaussian distirubiton
xx = linspace(-pi/2, pi/2,100);
vmy = circ_vmpdf(xx,0,8.74);
gaussy = normpdf(xx*180/pi,0,20);
vmy = vmy/sum(vmy);
gaussy = gaussy/sum(gaussy);


nsamples = 10000;
samples = circ_vmrnd(0,8.74,[nsamples]);
subplot(2,1,1); hist(samples*(180/pi),1000)
xlim([-100 100])

subplot(2,1,2); plot(xx*180/pi,vmy,xx*180/pi,gaussy);
legend('vm','gauss');
%% plotting sigma as a function of elipse reliaility

reliabilities = [0.15 0.4 0.6 0.8 0.95 0.999];
sigmas = [76.3 24.1 12.8 12.4 8.6 7.8];

plot(reliabilities,sigmas,'o');

%% plotting ellipse reliabilities

reliabilityNum = [0.6 0.7 0.8 0.9 0.95 0.99 0.995 0.999];

for i = 1:length(reliabilityNum);
    im = drawEllipse(300,reliabilityNum(i),45,1,0.4);
    subplot(2,4,i); imshow(im); colormap('gray')
end

%% testing stimuli
Screen('Preference', 'SkipSyncTests', 1);

% screen info (visual)
screenNumber = 1;% max(Screen('Screens'));       % use external screen if exists
[w, h] = Screen('WindowSize', screenNumber);  % screen resolution of smaller display
screenResolution = [w h];                 % screen resolution
screenCenter = screenResolution/2;       % screen center
screenDistance = 45;                      % distance between observer and screen (in cm)
screenAngle = 2*(180/pi)*(atan((prefs.screenWidth/2) / screenDistance)) ; % total visual angle of screen
screen_ppd = screenResolution(1) / screenAngle;  % pixels per degree
prefs.ellipseArea = prefs.ellipseArea * screen_ppd^2; % ellipse area (pixels)

% open screen
windowPtr = Screen('OpenWindow',screenNumber,prefs.grey,[],32,2);
Screen(windowPtr,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

linee = nan(2,1);
for i = 1:180;
    [linee(1), linee(2)] = lineCoord(100,i);
    xy = [linee -linee];
    linee
    Screen('DrawLines',windowPtr, xy, 4, [255 255 255] ,[w/2 h/2],1);
    Screen('Flip', windowPtr);
    pause(0.1);
end

sca;

%% draw a gaussian distribution
penwid = 10; 
nSamples = 100;
yy = -h*h/2*normpdf(linspace(-w,w,nSamples),0,w/3) + 0.8*h;
xx = linspace(0.2*w,0.8*w,nSamples);
%use FramePoly with a window, color, vector cordinates, and penwidth
Screen('FramePoly', windowPtr, prefs.white, [xx' yy'], penwid); 
Screen('DrawLine',windowPtr,prefs.grey,xx(1), yy(1), xx(end), yy(end),6.99);
Screen('Flip',windowPtr);
pause
sca


