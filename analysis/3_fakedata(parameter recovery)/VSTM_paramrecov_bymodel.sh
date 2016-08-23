#!/bin/bash
#PBS -l nodes=1:ppn=5
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=20GB
#PBS -m abe
#PBS -N VSTM_paramrecov_bymodel

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
subjids = cellfun(@(x) ['F' jobnum(1) x ],{'1','2','3','4','5','6','7','8','9','10'},'UniformOutput',false); % model and subjnum
model = str2num(jobnum(end)); % test model
nStartVals = 50;

fitparam_realdata(subjids,model, nStartVals)

EOF


