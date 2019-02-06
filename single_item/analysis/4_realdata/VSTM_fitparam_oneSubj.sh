#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=8GB
#PBS -m abe
#PBS -N VSTM_paramfit_singlesubj

index=${PBS_ARRAYID}
job=${PBS_JOBID}
walltime_lim=${PBS_WALLTIME}
script_name=${PBS_JOBNAME}
module purge
module load matlab

export MATLABPATH=/home/ay963/matlab-scripts
cat<<EOF | matlab -nodisplay
addpath('/home/ay963/job-scripts')
addpath(genpath('/home/ay963/matlab-scripts'))

blah = num2str($index);
modelname = str2num(blah(1));
subjidx = str2num(blah(2));
nIter = [blah(3:end) '0000'];

subjids = {'1','2','3','4','5'};
subjid = [subjids{subjidx} '_' nIter];
nStartVals = 50;

fitparam_realdata({subjid},modelname, nStartVals)

EOF
