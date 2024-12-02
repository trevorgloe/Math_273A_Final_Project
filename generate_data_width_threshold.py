## This file generates data to study the threshold width for the student and or teacher model 
# with different monomial activation functions
# 
# It saves the data into a fold called "m_thresh_data"

# inport nn files
from experiment import *
from monomial_neural_network import *
import matplotlib.pyplot as plt
import numpy as np
import os

## Create a function that will make data and train a neural network using a given number of data points and epochs
def test_training(n, d, epochs_reported, k_stud, k_teach, num_epochs,p,lr):
    # n is the number of data points
    # k is the hidden layer depth
    # M is the number of epochs

    # d = 5 # just fix the dimension of the data for now
    teacher_k = [k_teach] # single layer
    teacher_model = generate_teacher_model_noOutWeight(d, teacher_k,power=p) # use unit weights for these calculations
    # teacher_model = generate_teacher_model(d, teacher_k)
    print(teacher_model)

    # generate data
    data = generate_data(n, d, teacher_model)

    # create student
    student_k = [k_stud] # student model hidden layer sizes - 2 layers with increasing number of neurons
    student_model = generate_student_model_noOutWeight(d, student_k,power=p)
    # student_model = generate_student_model(d, k=student_k)
    print(student_model)
    # train the student
    student_model, losses = train(
        model = student_model, 
        x_train = data[0], 
        y_train= data[1], 
        num_epochs = num_epochs, 
        lr = lr,
        print_stuff=False,
        epochs_reported=epochs_reported
        )
    
    # print(student_model.layers[0].weight)
    # print(student_model.layers[2].weight)
    # print(teacher_model.layers[0].weight)
    # print(teacher_model.layers[2].weight)
    student_w = student_model.layers[0].weight.detach().numpy()
    teacher_w = teacher_model.layers[0].weight.detach().numpy()
    # return the final loss
    return losses, student_w, teacher_w

d = 4 # dimension of data
teach_stud = 0 # whether we will vary the width of the student of teacher model, 0=student, 1=teach
M = 30 # number of test runs that is averaged over for each value of m
num_pts = 1000 # number of data points used
epochs = 100000
m_min = 2
m_max = 4
m_step = 3
# mvec = np.arange(m_min, m_max, m_step)
mvec = [4,5,6,7,8,12,16,18]
# mvec = [2, 6]
k_fixed = 16 # fixed width for the teacher or student depending on which one we're varying
epochs_reported = 200 # how often is the data actually recorded 
p=3 # monomial power
lr = 1e-4 # learning rate

## create folder for saving data if it doesnt exist
if not os.path.exists("m_thresh_data"):
    os.mkdir("m_thresh_data")

## create a folder for the current run
if teach_stud==0:
    folder_name = "loss_values_p="+str(p)+"_d="+str(d)+"_var_stud_M="+str(M)+"n="+str(num_pts)+"epochs="+str(epochs)+"_fixed_k="+str(k_fixed)+"epochs_reported="+str(epochs_reported)+"_lr="+str(lr)
else:
    folder_name = "loss_values_p="+str(p)+"_d="+str(d)+"_var_teach_M="+str(M)+"n="+str(num_pts)+"epochs="+str(epochs)+"_fixed_k="+str(k_fixed)+"epochs_reported="+str(epochs_reported)+"_lr="+str(lr)

data_dir = os.path.join("m_thresh_data",folder_name)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)



# create empty lists for the data
all_student_w = []
all_teacher_w = []
all_loss = []

## loop over all the m values
for m in mvec:
    # create name for data
    this_m_folder_name = os.path.join(data_dir,"m="+str(m))
    print(this_m_folder_name)
    os.makedirs(this_m_folder_name)
    l_tot = np.zeros(int(np.floor(epochs/epochs_reported)))
    for i in range(M):
        # iterate over the number of times we want to take the average over
        print("run = "+str(i))
        if teach_stud==0:
            curr_l, stud_w, teach_w = test_training(n=num_pts, d=d, epochs_reported=epochs_reported, k_stud=m, k_teach=k_fixed,num_epochs=epochs,p=p,lr=lr)
        else:
            curr_l, stud_w, teach_w = test_training(n=num_pts, d=d, epochs_reported=epochs_reported, k_teach=m, k_stud=k_fixed,num_epochs=epochs,p=2,lr=lr)
        
        l_tot = l_tot + curr_l
        # save the data
        file_name = os.path.join(this_m_folder_name, "run="+str(i))
        np.save(file_name,curr_l)

    avg_l = l_tot / M # take the average

    all_loss.append(avg_l)
    all_student_w.append(stud_w)
    all_teacher_w.append(teach_w)

    


# print out the data to make sure the runs were reasonable
fig = plt.figure()
iters = np.arange(0,epochs,epochs_reported)
# print(iters)
# print(all_loss[1])
for idx,m in enumerate(mvec):
    plt.semilogy(iters,all_loss[idx],label='m='+str(m))

plt.legend()
plt.show()

