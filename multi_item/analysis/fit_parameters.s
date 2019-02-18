#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=fit_parameters
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=o_%a.out

module purge
module load matlab/2016b

cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection/helper_functions'))
addpath(genpath('/home/ay963/changedetection/multi_item'))

% get indices of data and model
blah = num2str($SLURM_ARRAY_TASK_ID);
modelidx = str2double(blah(3:4));
subjidx = str2double(blah(1));
conditionidx = str2double(blah(2));

% model fitting settings
runmax = 50;
runlist = 2:50;
% runlist = runlistidx:5:(45+runlistidx);
nSamples = 200;

% subject and model info
subjVec = {'POO','METEST'};
conditionVec = {'Ellipse','Line'};
modelMat = ...
   [1 1 1;  1 2 1; 1 3 1; ... % VVO, VFO, VSO
    1 1 2;  1 2 2; 1 3 2; ... % VVM, VFM, VSM
            2 2 1; 2 3 1; ... % FVO, FFO, FSO
            2 2 2; 2 3 2];    % FVM, FFM, FSM
subjid = subjVec{subjidx};
model = modelMat(modelidx,:);
condition = conditionVec{conditionidx};

% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))

find_ML_parameters(data,model,runlist,runmax,nSamples)

EOF
