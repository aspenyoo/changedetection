#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=fit_parameters
#SBATCH --mail-type=END
#SBATCH --mail-user=aspen.yoo@nyu.edu
#SBATCH --output=fp_%a.out

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

load('jobsettings.mat')
subjid = subjidCell{idx};
model = modelCell{idx};
condition = conditionCell{idx};
runlist = runlistCell{idx};

% load data
load(sprintf('../data/fitting_data/%s_%s_simple.mat',subjid,condition))

find_ML_parameters(data,model,runlist,runmax,nSamples)

blah

EOF
