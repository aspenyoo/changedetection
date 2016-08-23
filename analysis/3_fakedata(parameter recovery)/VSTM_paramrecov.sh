#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=8GB
#PBS -m abe
#PBS -N VSTM_pparamfit

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

jobnum = num2str($index);
subjid = {['F' jobnum(1:end-1)]}; % model and subjnum
model = str2num(jobnum(end)); % test model
nStartVals = 50;

fitparam_realdata(subjid,model, nStartVals)

EOF


