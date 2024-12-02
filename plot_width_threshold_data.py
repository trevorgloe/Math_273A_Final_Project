## This file loads in data from the m_thresh_data folder and plots it
# I'm putting this into a .py file to make it more reproducable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

def organize_runs(run_list):
    # oragnize the runs in accending order in m
    new_list = []
    m_ints = []
    for name in run_list:
        temp = name.split("=")
        int_numpy = temp[1]
        temp2 = int_numpy.split(".")
        just_int = temp2[0]
        m_ints.append(int(just_int))

    sorted_ms = m_ints.copy()
    sorted_ms.sort()

    for sorted_int in sorted_ms:
        new_list.append("m="+str(sorted_int))

    return new_list

run_name = "loss_values_p=3_d=4_var_stud_M=30n=1000epochs=100000_fixed_k=16epochs_reported=200_lr=0.0001"
folder_name = os.path.join("m_thresh_data",run_name)
data_run_names = os.listdir(folder_name)
data_run_names = organize_runs(data_run_names)
print(data_run_names)
num_ms = len(data_run_names)
epochs_reported = 200
epochs = 100000

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
plt.grid(True)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.show()