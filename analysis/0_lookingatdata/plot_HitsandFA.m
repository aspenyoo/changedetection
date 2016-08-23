% plotting hits and FAs for each reliability
close all
clear all
clc

experimenttype = 'Detection';
% subjids = {'1','3','4','EL','ND'};
nSubj = 5;
nCond = 5;
model = 7;

load(sprintf('paramfit_model%d_04292016.mat',model));

pHit = cell(1,nCond); pFA = cell(1,nCond);
model_pHit = cell(1,nCond); model_pFA = cell(1,nCond);
for isubj = 1:nSubj;
    % data hits and FAs
    subjid = subjids{isubj};
    data = concatdata(subjid, experimenttype);
    Xdet = conditionSeparator(data,1);
    
    pHit = cellfun(@(x,y) [x; mean(y(y(:,1)~=0,2)==0)],pHit,Xdet,'UniformOutput',false);
    pFA = cellfun(@(x,y) [x; mean(y(y(:,1)==0,2)==0)],pFA,Xdet,'UniformOutput',false);
    
    % theta
    theta = bestFitParam(isubj,:);
    % adjusting theta to match log stuff
    switch model
        case 1 % Optimal model, only free noise and lapse
            logflag = logical([ones(1,nCond) 0]);
        case 2 % Optimal model + free prior on (C=1)
            logflag = logical([ones(1,nCond) 0 0]);
        case 3 % Optimal model + free prior on C + free prior on S
            logflag = logical([ones(1,nCond) 0 1 0]);
        case 4 % Optimal model + free prior on S
            logflag = logical([ones(1,nCond+1) 0]);
        case 5 % Fixed criterion model: free noise, sigmatilde, lapse
            logflag = logical([ones(1,nCond+1) 0]);
        case 6 % Super-free model: free noise and full free sigmatilde and lapse
            logflag = logical([ones(1,2*nCond) 0]);
        case 7 % linear heuristic model: free noise, low and high kcommon, lapse
            logflag = logical([ones(1,nCond) zeros(1,3)]);
    end
    theta(logflag) = log(theta(logflag));
    
    % model hits and FAs
    model_Xdet = cellfun(@(x) [x(:,1) ones(size(x(:,1)))],Xdet,'UniformOutput',false);
    [~,presp] = loglike(model_Xdet, model, theta);
    model_Xdet = cellfun(@(x,y) [x(:,1) binornd(1,y)], model_Xdet,presp,'UniformOutput',false);
    
    model_pHit = cellfun(@(x,y) [x; mean(y(y(:,1)~=0,2)==0)],model_pHit,model_Xdet,'UniformOutput',false);
    model_pFA = cellfun(@(x,y) [x; mean(y(y(:,1)==0,2)==0)],model_pFA,model_Xdet,'UniformOutput',false); 
end

% making all of them matrices
pHit = cell2mat(pHit);
pFA = cell2mat(pFA);
model_pHit = cell2mat(model_pHit);
model_pFA = cell2mat(model_pFA);

% mean and sem
mean_pHit = mean(pHit);
sem_pHit = std(pHit)/sqrt(nSubj);
mean_pFA = mean(pFA);
sem_pFA = std(pFA)/sqrt(nSubj);
mean_model_pHit = mean(model_pHit);
sem_model_pHit = std(model_pHit)/sqrt(nSubj);
mean_model_pFA = mean(model_pFA);
sem_model_pFA = std(model_pFA)/sqrt(nSubj);

% plot
pink = aspencolors('dustyrose');
blue = [0 0 1];
figure;
plot_summaryfit(1:5,mean_pHit,sem_pHit,mean_model_pHit,sem_model_pHit,pink,pink);
hold on;
plot_summaryfit(1:5,mean_pFA,sem_pFA,mean_model_pFA,sem_model_pFA,blue,blue);
legend('data hits','model hits','data FAs','model FAs')
set(gca,'Xtick',1:5)
xlabel('reliability condition')
ylabel('proportion response')
