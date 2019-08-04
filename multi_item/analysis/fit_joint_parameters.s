#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=fit_joint_parameters
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=jo_%a.out

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
blah = num2str($SLURM_ARRAY_TASK_ID);
subjidx = str2double(blah(1));
modelidx = str2double(blah(2:3));
runlist = 1:20;

subjidVec = {'S15','S16','S17','S19','S20','S23'};
% subjidVec = {'S02','S03','S06','S07','S08','S10','S11','S14','POO','METEST'};
subjid = subjidVec{subjidx};
modelMat = ...
    [1 1 1;  1 2 1; 1 3 1; ...  % V_O model variants
     1 1 2;  1 2 2; 1 3 2; ...  % V_M model variants
             2 2 1; 2 3 1; ...  % F_O model variants
             2 2 2; 2 3 2];     % F_M model variants
model = modelMat(modelidx,:);


% load data and save as appropriate variables
load(sprintf('data/fitting_data/%s_Ellipse_simple.mat',subjid))
data_E = data;
load(sprintf('data/fitting_data/%s_Line_simple.mat',subjid))
data_L = data;

find_joint_ML_parameters(data_E,data_L,model,runlist,runmax,nSamples)

EOF
