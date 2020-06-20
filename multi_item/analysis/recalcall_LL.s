#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=fit_parameters
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=all_llcalc_%a.out

module purge
module load matlab/2016b

cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection/helper_functions'))
addpath(genpath('/home/ay963/changedetection/multi_item'))

% fixed model fitting settings
condition = 'Line';

% fitting settings (determined by index)
idx = $SLURM_ARRAY_TASK_ID;

load('modelfittingsettings.mat')

[imodel,isubj] = ind2sub([nModels nSubjs],idx);
model = modelMat(imodel,:);
subjid = subjidVec{isubj}

load(sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,subjid,...
condition,model(1),model(2),model(3),model(4)));

options_ibs = ibslike('defaults');
options_ibs.Vectorized = 'on';
options_ibs.MaxIter = 10000;
options_ibs.Nreps=1000;
logflag = [];
        
% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))

% data in ibs format
dMat = data.Delta;
rels = unique(data.rel);
blah = data.rel;
for irel = 1:length(rels)
    blah(blah == rels(irel)) = irel;
end
dMat = [dMat blah];

nRuns = size(bfp,1);

LLVec_old = LLVec;
LLVec = nan(1,nRuns);


for irun = 1:nRuns
    fun = @(x,y) fun_LL(x,y,model,condition,logflag);
    [LLVec(irun), LLvarVec(irun)]= ibslike(fun,x,data.resp,dMat,options_ibs);
end

save(sprintf('fits/%s/subj%s_%s_model%d%d%d%d.mat',condition,subjid,...
condition,model(1),model(2),model(3),model(4)),...
'bfp','completedruns','LLVec','LLVec_old','LLvarVec');



EOF
