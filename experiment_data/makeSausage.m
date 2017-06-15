function sausage = makeSausage(imSize, s_center, s_length, contrast, sigma, spatialfreq)

% stuff you can change
if nargin < 1; imSize = 128; end
if nargin < 2 s_center = 0; end         % center orientation (radians)
if nargin < 3; s_length = 0; end        % length of sausage (radians)
if nargin < 4; contrast = 1; end        % contrast: 0 to 1
if nargin < 5; sigma = 2; end           % gaussian envelope
if nargin < 6; spatialfreq = 7; end     % how many cycles in entire image

% image size
xx = (-imSize/2):imSize/2-1; % image size
[X,Y] = meshgrid(xx,xx);
nsamps = length(xx);

% thetas used for sausage
s_center = pi/2 - s_center; % making 0 the positive x-axis and rotate counterclockwise
thetaVec = linspace(s_center-s_length/2, s_center+s_length/2,50);
nThetas = length(thetaVec);

% make sausage in freq space
freqSpace = zeros(nsamps^2,1);
% for each theta in sausage
for itheta = 1:nThetas
    theta = thetaVec(itheta);
    [u,v] = pol2cart(theta,spatialfreq);
    
    % gaussian window
    freqSpace = freqSpace + mvnpdf([X(:) Y(:)],[u v],[sigma sigma]);
    freqSpace = freqSpace + mvnpdf([X(:) Y(:)],[-u -v],[sigma sigma]);
end
freqSpace = reshape(freqSpace,nsamps,nsamps).*100;

% make every odd complex and every even real
idx = logical(mod(1:nsamps,2));
IDX = reshape(mod(1:nsamps^2,2),nsamps,nsamps);
IDX(:,idx) = repmat(logical(1-idx)',1,imSize/2);
IDX = logical(IDX);

% freqSpace(IDX) = freqSpace(IDX)./sqrt(2) - freqSpace(IDX).*i./sqrt(2);
% freqSpace(~IDX) = -freqSpace(~IDX)./sqrt(2) + freqSpace(~IDX).*i./sqrt(2);

% making actual gabor sausage stimulus
stimSpace = ifft2(freqSpace); % 2D inverse fourier transform
stimSpace = ifftshift(stimSpace,1); % shifting first dimension
stimSpace = ifftshift(stimSpace,2); % shifting second dimension
sausage = real(stimSpace);
sausage(IDX) = -sausage(IDX);

% contrast stuff
sausage = sausage./max(abs([min(sausage(:)) max(sausage(:))])); % make it relative to -1 to 1 scale
sausage = sausage.*127.5*contrast; % scale by contrast
sausage = round(sausage + 128); % scale: 0: 255

% % ====== MAKE FIGURE =======
% % stimulus in frequency space
% figure('units','normalized','position',[.1 .1 .6 .4])
% subplot(1,2,1);
% imagesc(abs(freqSpace));
% set(gca,'Xtick',[],'YTick',[])
% colormap('gray')
% 
% % stimulus in x,y space
% subplot(1,2,2)
% % figure;
% imagesc(sausage);%,[0 255]);
% set(gca,'Xtick',[],'YTick',[])
% colormap('gray')


%%
% 
% figure('units','normalized','position',[.1 .1 .6 .4])
% rng = 1:nsamps;
% 
% subplot(1,2,1);
% imagesc(real(stimSpace(rng,rng)))
% 
% subplot(1,2,2);
% imagesc(imag(stimSpace(rng,rng)))


%% 
% clear all
% close all
% 
% spatialfreq = 2; % 
% imSize = 10; % pixels
% nSamps = 100;
% xx = linspace(0,imSize,nSamps);
% % 
% % 
% % sigma = 0.5;
% % x = normpdf(xx,spatialfreq,sigma);
% % 
% % figure; plot(x)
% % figure; plot(real(ifft2(x)));
% 
% 
% [X,Y] = meshgrid(xx,xx);
% % Y = flipud(Y);
% 
% theta = pi/4; % starting from positive x axis
% 
% % change from polar to cartesian coordinates
% [x,y] = pol2cart(theta, spatialfreq);
% 
% % make gaussian around x
% sigma = 0.5;
% blah = mvnpdf([X(:) Y(:)],[x,y],[sigma sigma]);
% blah = reshape(blah,nSamps,nSamps);
% blah = blah + fliplr(flipud(blah));
% figure; imagesc(blah);
% 
% 
% figure;
% imagesc(reshape(blah,nSamps,nSamps))
% 
% 
% figure;
% bleh = ifft2(blah);
% imagesc(reshape(real(bleh),nSamps,nSamps))
% 
% 
% %% impulse to sinusoid
% 
% xx = 0:100;
% nX = length(xx);
% idx = 3;
% blah = zeros(1,length(xx));
% blah(idx+1) = 5;
% blah(end-(idx-1)) = 5;
% 
% figure; subplot(2,1,1);
% stem(xx,blah,'k')
% defaultplot
% % set(gca,'YTick',[])
% 
% yy = ifftshift(ifft(blah));
% 
% subplot(2,1,2)
% plot(yy,'k-')
% defaultplot
% % set(gca,'YTick',[])
% 
% %% gaussian to sinusoid
% 
% xx = 0:100;
% nX = length(xx);
% idx = 2;
% idx1 = idx+1;
% idx2 = length(xx)-idx+1;
% sigma = 1;
% 
% blah = normpdf(xx,idx1,sigma);
% blah = blah + normpdf(xx,idx2,sigma);
% 
% figure; subplot(2,1,1);
% stem(blah)
% 
% yy = ifftshift(ifft(blah));
% idxneg = (real(yy) > 0) & (imag(yy) < 0);
% idxneg = idxneg + (real(yy) < 0) & (imag(yy) > 0);
% 
% % yy = abs(yy);
% % yy(idxneg) = -yy(idxneg);
% 
% 
% subplot(2,1,2)
% plot(xx,yy,'o-')
% 
% 
% %% 
% N = 100;
% n = 0:100;
% 
% f1 = sin(2*pi*3*n/N);
% 
% f2 = (fft(f1));
% f2 = abs(f2);
% f2(f2<10) = 0;
% 
% f3 = (ifft(f2));
% % f3 = f3 + 2i;
% % f3 = real(f3);
% 
% figure;
% subplot(3,1,1)
% plot(n,f1,'o-')
% 
% subplot(3,1,2)
% stem(f2)
% 
% subplot(3,1,3)
% plot(n,f3,'o-')
% 
% 
