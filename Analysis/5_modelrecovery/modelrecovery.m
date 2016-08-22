% function modelrecovery

modelVec = []; % vector of models being compared
nModels = length(modelVec);
nSubj = 3;

for itruemodel = 1:nModel;
    truemodel = modelVec(itruemodel);
    
    % getting average estimates for subjects
    filename = ['M' num2str(truemodel) '_fits.mat'];
    load(filename)
    mean_theta = mean(thetaVec);
    sem_theta = std(thetaVec)/sqrt(size(thetaVec,1));
    nParams = length(mean_theta);
    
    % simulate data with each model
    for isubj = 1:nSubj;
        truetheta{itruemodel}(isubj,:) = mean_theta + randn(1,nParams).*sem_theta;
        
        if datafittype(1);
            data = concatdata(num2str(isubj),'Discrim');
            Xdsc = conditionSeparator(data,isgaussprior);
        else
            Xdsc = [];
        end
        
        if datafittype(2);
            data = concatdata(num2str(i),'Detection');
            Xdet = conditionSeparator(data,isgaussprior);
        else
            Xdet = [];
        end
        
        sigma = theta(i,1:nCond);
        lapserate = theta(i,end);
        mu = [];
        sigmatilde = [];
        
        YNuniform = 1-isgaussprior;
        switch model
            case 1
                prior = [0.5, 20]; % SD of uniform distribution
            case 2
                prior = [theta(end-1), 20]; % gaussian prior
            case 3
                prior = [theta(end-2), exp(theta(end-1))];
            case 4
                prior = [theta(end-1), -45]; % Half-length of uniform distribution
            case 5
                sigmatilde = theta(nCond+1);
                if (YNuniform)
                    prior = [0.5, -45];
                else
                    prior = [0.5, 20];
                end
            case 6
                sigmatilde = theta(nCond+1:2*nCond);
                if (YNuniform)
                    prior = [0.5, -45];
                else
                    prior = [0.5, 20];
                end
        end
        
        [loglike, logprior, P_resp] = datalikelihood(Xdsc,Xdet,mu,sigma,sigmatilde,prior,lapserate);
        
        for ifitmodel = 1:nModel;
            
            % fit data with each model
            
            
            
        end
    end
    
end