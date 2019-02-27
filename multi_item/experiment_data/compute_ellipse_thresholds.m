function th = compute_ellipse_thresholds(TrialMat,makeplot)

% if the threshold block wasn't finished
if sum(TrialMat(:,4)>0) ~= size(TrialMat,1)
    error('Threshold block not complete! Remove unfinished trials or finish block.')
end

% get the eccentricity values
X = unique(TrialMat(:,39));
Y = zeros(length(X),1);
for ii = 1:length(X)
    curr_ecc_trials = TrialMat(TrialMat(:,39)==X(ii),:);
    Y(ii) = sum((curr_ecc_trials(:,1)~=0)==curr_ecc_trials(:,2))/size(curr_ecc_trials,1);
end

% fit with different starting values
nEvals = 10;
fitpars = nan(nEvals,3);
fval = nan(1,nEvals);
for ival = 1:nEvals;
    [fitpars(ival,:), fval(ival)] = fminsearch(@(fitpars) fitfun(fitpars,X,Y),rand(1,3));
end
idx_min = find(fval == min(fval),1,'first');
fitpars = fitpars(idx_min,:);

X_fit = 0:.001:1;
Y_fit = 0.5 + fitpars(3) * normcdf(X_fit,fitpars(1),fitpars(2));
Y_fit = Y_fit + randn(size(Y_fit)).*1e-6;
% Y_fit = normrnd(Y_fit,.000001);   % to avoid "Values should be distinct" error in interp1
th  = interp1(Y_fit,X_fit,.65);

% if plotting
if (makeplot)

    figure;
    set(gca,'FontSize',14);
    plot(X,Y,'rx','LineWidth',2,'MarkerSize',12);
    set(gca,'FontSize',12);
    hold on;
    plot(X_fit,Y_fit,'k-','LineWidth',2)
    legend({'data','cdf fit'},'Location','NorthWest');
    xlim([0 1]);
    text(.05,.85,['\epsilon_{low} = ' num2str(th,2)]);
    text(.05,.79,['ceiling = ' num2str(.5+fitpars(3),2)]);
    xlabel('Eccentricity (\epsilon)');
    ylabel('Proportion correct');
    ylim([min(Y) 1.05]);
    ylim([.4 1]);       
end

% if the asymptotic performance isn't higher than 75%, throw warning
if (.5+fitpars(3)) < 0.75    
    warning('Ceiling performance did not reach 75%, subject should redo threshold block!')
end

% helper function
function sse = fitfun(pars,uCont,Y)
pars(3)=min(.5,pars(3));
sse = sum( ((0.5 + pars(3) * normcdf(uCont,pars(1),pars(2))) - Y).^2) ;