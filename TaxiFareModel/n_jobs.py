"""what is n_jobs"""

#https://towardsdatascience.com/understanding-the-n-jobs-parameter-to-speedup-scikit-learn-classification-26e3d1220c28

import os
    
n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

"""n_jobs is literally: number of jobs. 
How many CPUs (workers) are going to work for a GridSearch or a model"""

