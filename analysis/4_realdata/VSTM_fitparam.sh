#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=6GB
#PBS -m abe
#PBS -N VSTM_paramfit

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

subjids = {'1','2','3','4','5'};
blah = num2str($index);
modelname = str2double(blah(1));
subjids = cellfun(@(x) [x '_' blah(2:end) '0000'],subjids,'UniformOutput',false);
nStartVals = 50;

fitparam_realdata(subjids,modelname, nStartVals)

EOF


