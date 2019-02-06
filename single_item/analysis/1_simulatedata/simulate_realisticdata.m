function [Xdet] = simulate_realisticdata(model,subjidz)

nSubj = length(subjidz);

% load bestFitParam for the model
files = dir(['analysis/4_realdata/fits/paramfit_model' num2str(model) '_04292016.mat']);
if length(files) == 1;
    load(files.name);
    mean_bfp = mean(bestFitParam);
    cov_bfp = cov(bestFitParam);
    nParams = length(mean_bfp);
else
    error('problems with loading fitted parameters')
end

nCond = 5;
switch model
    case 1 % Optimal model, only free noise and lapse
        lb = [(zeros(1,nCond)) 0];
        ub = [(200*ones(1,nCond)) 1];
    case 2 % Optimal model + free prior on (C=1)
        lb = [(zeros(1,nCond)) 0 0];
        ub = [(200*ones(1,nCond)) 1 1];
    case 3 % Optimal model + free prior on C + free prior on S
        lb = [(zeros(1,nCond)) 0 (0) 0];
        ub = [(200*ones(1,nCond)) 1 (200) 1];
    case 5 % Fixed criterion model: free noise, sigmatilde, lapse
        lb = [(zeros(1,nCond+1)) 0];
        ub = [(200*ones(1,nCond+1)) 1];
    case 6 % Super-free model: free noise and full free sigmatilde and lapse
        lb = [(zeros(1,2*nCond)) 0];
        ub = [(200*ones(1,2*nCond)) 1];
    case 7 % linear heuristic
        lb = [zeros(1,nCond) zeros(1,2) 0];
        ub = [200*ones(1,nCond) 100*ones(1,2) 1];
end

for isubj = 1:nSubj;
    subjid = subjidz{isubj};
    
    theta = mvnrnd(mean_bfp,cov_bfp);
    
    % making sure its in the bounds
    while sum(theta < lb) + sum(theta > ub);
        theta = mvnrnd(mean_bfp,cov_bfp);
    end
    
    [Xdet] = simulateresp(model, theta);
    
    % save
    if nargout < 1;
        subjid = ['F' num2str(model) subjid];
        save(sprintf('analysis/3_fakedata/fakedata/fakedata_subj%s.mat',...
           subjid),'Xdet','model','subjid','theta');
    end
end
