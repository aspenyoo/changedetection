This folder contains the functions used to fit and compare models. 

## File/Directory 
All files and file directories are described below. More detailed descriptions can be found in each file. On MATLAB, you can type "help function" or "doc function" to read this descriptions, if you don't want to simply open the file. 

- fits/: directory containing fits for subjects and models
- calculate_joint_LL.m: calculates the likelihood of data given model for data in both Ellipse and Line conditions
- calculate_LL.m: calculates the likelihood of data given model for one condition (Line or Ellipse)
- clusterfittingsettings.mat: matrix data file used to fit models on HPC (probably not useful for you)
- find_joint_ML_parameters.m: finds the ML parameter estimates for all data (Ellipse and Line conditions)
- find_ML_parameters.m: finds the ML parameter estimates for just one condition (Ellipse or Line)
- fit_joint_parameters: bash script used to fit joint parameters on HPC (may not be useful for you)
- fit_parameters.s: bash script used to fit single-condition parameters on HPC (may not be useful for you)
- getFittingSettings.m: contains settings for parameter estimation
- make_cdf_table.m: used to sample kappas faster in calculate_LL.m
- plot_psychometric_fn.m: bins and plots data
- qinterp1.m: interpolation function used in calculate_LL.m
- simulate_data.m: simulates data for some model and parameter combination

## Difference between this code vs. code used in Keshvari et al., (2012)
The models are the same as in [Keshvari et al. (2012)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040216), but the code to fit is different. Here, we use 1) maximum-likelihood to obtain parameter estimates and 2) AICc and BIC for model comparison. In Keshvari et al., (2012), they 1) computed the likelihood of parameters over a grid search and 2) completed a Bayesian model comparison using these results. Code used to analyze data in the Keshvari et al., (2012) paper can be found here.  

This folder is (as of February 6, 2019), an exact copy of the analysis script use to fit and compare models in Keshvari et al., 2012. You can download the original folder [here](https://www.dropbox.com/sh/y11jfw11ifw99mx/AADdK1O64KgW5SNnLc7Wb9eHa?dl=0 "dropbox link"). 

