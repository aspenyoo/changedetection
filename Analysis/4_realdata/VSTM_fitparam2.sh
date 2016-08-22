#!/bin/bash
#PBS -l nodes=1:ppn=3
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=18GB
#PBS -m abe
#PBS -N VSTM_paramfit2

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

modelname = $index;
nStartVals = 50;

fitparam_realdata({'NN31','NN32','NN33'},modelname, nStartVals)

EOF

