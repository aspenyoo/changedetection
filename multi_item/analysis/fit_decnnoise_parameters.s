#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=fit_decnoise
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=o_%a.out

module purge
module load matlab/2016b

cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection/helper_functions'))
addpath(genpath('/home/ay963/changedetection/multi_item'))

% fixed model fitting settings
runmax = 20;
nSamples = [500 1000];

% fitting settings (determined by index)
idx = $SLURM_ARRAY_TASK_ID;
runlist = 1:10;

subjidVec = {'S02','S03','S06','S08','S10','S11','S14','S15','S16','S17','S19','S20','S23'}; % all real full subjects
modelMat = ...
   [1 1 1 0;  1 2 1 0; 1 3 1 0; 1 4 1 0; ...  % V_O model variants --|
    1 1 2 0;  1 2 2 0; 1 3 2 0; 1 4 2 0; ...  % V_M model variants   |  no decision
              2 2 1 0; 2 3 1 0; 2 4 1 0; ...  % F_O model variants   |     noise
              2 2 2 0; 2 3 2 0; 2 4 2 0; ...  % F_M model variants __|
    1 1 1 1;  1 2 1 1; 1 3 1 1; 1 4 1 1; ...  % V_O model variants --|
    1 1 2 1;  1 2 2 1; 1 3 2 1; 1 4 2 1; ...  % V_M model variants   |  local decision
              2 2 1 1; 2 3 1 1; 2 4 1 1; ...  % F_O model variants   |      noise
              2 2 2 1; 2 3 2 1; 2 4 2 1; ...  % F_M model variants __|
    1 1 1 2;  1 2 1 2; 1 3 1 2; 1 4 1 2; ...  % V_O model variants --|
    1 1 2 2;  1 2 2 2; 1 3 2 2; 1 4 2 2; ...  % V_M model variants   |  global decision
              2 2 1 2; 2 3 1 2; 2 4 1 2; ...  % F_O model variants   |      noise
              2 2 2 2; 2 3 2 2; 2 4 2 2];     % F_M model variants __|
conditionVec = {'Ellipse','Line'};

nSubjs = length(subjidVec);
nModels = size(modelMat,1);
nConds = length(conditionVec);

[isubj,imodel,icond] = ind2sub([nSubjs nModels nConds],idx);
subjid = subjidVec{isubj};
model = modelMat(imodel,:);
condition = conditionVec{icond};

% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))

load(sprintf('fits/%s/bfp_%s.mat',condition,condition))
idx = mod(imodel,14);
if (idx == 0); idx = 14; end
x0 = [bfpMat{idx}(isubj,:) 0];

find_ML_parameters(data,model,runlist,runmax,nSamples,x0)

blah

EOF
