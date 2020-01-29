#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=fp2
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=fp2_%a.out

module purge
module load matlab/2016b

cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection/helper_functions'))
addpath(genpath('/home/ay963/changedetection/multi_item'))

% fixed model fitting settings
condition = 'Line';
runlist = 1:20;
runmax = 20;

% subject/model info
subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14','S15','S16','S17','S19','S20','S23'}; % all real full subjects
modelMat = ...
   [1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ...  % V_Ol model variants
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ...  % V_Ml model variants
              2 2 1 1; 2 3 1 1; 2 4 1 1; ...  % F_Ol model variants
              2 2 2 1; 2 3 2 1; 2 4 2 1; ...  % F_Ml model variants
    1 1 1 2;  1 2 1 2; 1 3 1 2; 1 4 1 2; ...  % V_Og model variants
    1 1 2 2;  1 2 2 2; 1 3 2 2; 1 4 2 2; ...  % V_Mg model variants
              2 2 1 2; 2 3 1 2; 2 4 1 2; ...  % F_Og model variants
              2 2 2 2; 2 3 2 2; 2 4 2 2];     % F_Mg model variants
nSubjs = length(subjidVec);
nModels = size(modelMat,1);

% fitting settings (determined by index)
idx = $SLURM_ARRAY_TASK_ID;
[isubj,imodel] = ind2sub([nSubjs nModels],idx);
subjid = subjidVec{isubj};
model = modelMat(imodel,:);

% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))
for irun = runlist
    try
        find_ML_parameters(data,model,irun,runmax,nSamples)
    end
end
blah

EOF