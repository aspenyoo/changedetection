#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
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

% fixed model fitting settings
runmax = 20;
nSamples = [50 1000];

% fitting settings (determined by index)
idx = num2str($SLURM_ARRAY_TASK_ID);
isubj = str2double(idx(1));
imodel = str2double(idx(2:3));
condition = 'Ellipse';
runlist = 1:20;

subjidVec = {'S15','S16','S17','S19','S20','S23'};
% subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14','S04'};
% subjidVec = {'S91','S92','S93','S94','S95','S96','S97','S98','S99'};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
subjid = subjidVec{isubj};
model = modelMat(imodel,:);

% settings = clustersettings{idx};
% subjid = settings.subjid
% model = settings.model
% condition = settings.condition
% runlist = settings.runlist

% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))

find_ML_parameters(data,model,runlist,runmax,nSamples)

blah

EOF
