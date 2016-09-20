#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=4GB
#PBS -m abe
#PBS -N paramrecov_CD2

index=${PBS_ARRAYID}
job=${PBS_JOBID}
walltime_lim=${PBS_WALLTIME}
script_name=${PBS_JOBNAME}
module purge
module load matlab

export MATLABPATH=/home/ay963/matlab-scripts
cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/changedetection'))

subjids = {'6','7','8','9','10'};
nStartVals = 50;
blah = num2str($index);
testmodel = str2double(blah(1));
truemodel = str2double(blah(2));

fitparam_fakedata(subjids,testmodel, truemodel, nStartVals)

EOF


