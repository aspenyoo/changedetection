#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=fit_parameters
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=llcalc_e_%a.out

module purge
module load matlab/2016b

cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection/helper_functions'))
addpath(genpath('/home/ay963/changedetection/multi_item'))

% fixed model fitting settings
condition = 'Ellipse';

% fitting settings (determined by index)
idx = $SLURM_ARRAY_TASK_ID;

load(sprintf('fits/%s/bfp_%s.mat',condition,condition));
nModels = size(modelMat,1);
nSubjs = length(subjidVec);

[imodel,isubj] = ind2sub([nModels nSubjs],idx);
model = modelMat(imodel,:);
subjid = subjidVec{isubj}

options_ibs = ibslike('defaults');
options_ibs.Vectorized = 'on';
options_ibs.MaxIter = 10000;
options_ibs.Nreps=1000;
logflag = [];

x = bfpMat{imodel}(isubj,:);
        
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

fun = @(x,y) fun_LL(x,y,model,condition,logflag);
[LL, LLvar]= ibslike(fun,x,data.resp,dMat,options_ibs);

save(sprintf('fits/recalcLL_%s_imodel%d_isubj%d.mat',condition,imodel,isubj),'LL','LLvar')


EOF
