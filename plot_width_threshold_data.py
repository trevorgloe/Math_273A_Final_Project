## This file loads in data from the m_thresh_data folder and plots it
# I'm putting this into a .py file to make it more reproducable

import numpy as np
import matplotlib.pyplot as plt
import os

run_name = "loss_values_p=2_d=5_var_stud_M=10n=100epochs=20000_fixed_k=12epochs_reported=100_lr=0.001"
folder_name = os.path.join("m_thresh_data",run_name)
data_run_names = os.listdir(folder_name)
num_ms = len(data_run_names)
epochs_reported = 100
epochs = 20000

avg_losses = []
for run_name in data_run_names:
    # get all the files for that run and average all the losses
    run_folder_name = os.path.join(folder_name,run_name)
    same_m_names = os.listdir(run_folder_name)
    M=len(same_m_names)

    # get the first one to allocate the array
    print("Loading first run...")
    print(os.path.join(run_folder_name,same_m_names[0]))
    total_losses = np.load(os.path.join(run_folder_name,same_m_names[0]))

    for j in range(M-1):
        print("Loading file ")
        print(os.path.join(run_folder_name,same_m_names[j+1]))
        total_losses = total_losses + np.load(os.path.join(run_folder_name,same_m_names[j+1]))

    # take the average
    avg_l = 1/M * total_losses
    avg_losses.append(avg_l)
    

## make the plots
iters = np.arange(0,epochs,epochs_reported)
loss_plot_fig = plt.figure()
for i in range(num_ms):
    l = avg_losses[i]
    m_str = data_run_names[i]

    plt.semilogy(iters, l,label=m_str)

plt.xlabel("Iterations")
plt.ylabel("Empirical Loss")
plt.legend()
plt.show()