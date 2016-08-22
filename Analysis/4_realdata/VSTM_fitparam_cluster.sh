#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=4GB
#PBS -m abe
#PBS -N VSTM_paramfit_cluster

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

jobnum = $index;
nStartVals = 50;

fitparam_cluster(jobnum, nStartVals)

EOF


