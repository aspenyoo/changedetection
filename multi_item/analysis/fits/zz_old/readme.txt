June 3, 2019

these files were all corresponding to a buggy calculate_LL, which calculated p_C_hat = 1 when log(d)>1. 

However, it should be either

p_C_hat = 1 when d>1

pr 

p_C_hat = 1 when log(d) > log(1)

