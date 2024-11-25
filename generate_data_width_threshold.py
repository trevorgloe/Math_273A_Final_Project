## This file generates data to study the threshold width for the student and or teacher model 
# with different monomial activation functions
# 
# It saves the data into a fold called "m_thresh_data"

# inport nn files
from experiment import *
from monomial_neural_network import *
import matplotlib.pyplot as plt
import numpy as np

d = 5 # dimension of data
teach_stud = 0 # whether we will vary the width of the student of teacher model, 0=student, 1=teach
M = 5 # number of test runs that is averaged over for each value of m
num_pts = 1000 # number of data points used
epochs = 200000





## Create a function that will make data and train a neural network using a given number of data points and epochs
def test_training(n, k_stud, k_teach, M):
    # n is the number of data points
    # k is the hidden layer depth
    # M is the number of epochs

    # d = 5 # just fix the dimension of the data for now
    teacher_k = [k_teach] # single layer
    teacher_model = generate_teacher_model_noOutWeight(d, teacher_k) # use unit weights for these calculations
    # teacher_model = generate_teacher_model(d, teacher_k)
    print(teacher_model)

    # generate data
    data = generate_data(n, d, teacher_model)

    # create student
    student_k = [k_stud] # student model hidden layer sizes - 2 layers with increasing number of neurons
    student_model = generate_student_model_noOutWeight(d, student_k)
    # student_model = generate_student_model(d, k=student_k)

    # train the student
    student_model, losses = train(
        model = student_model, 
        x_train = data[0], 
        y_train= data[1], 
        num_epochs = M, 
        lr = 0.2e-4,
        print_stuff=False
        )
    
    # print(student_model.layers[0].weight)
    # print(student_model.layers[2].weight)
    # print(teacher_model.layers[0].weight)
    # print(teacher_model.layers[2].weight)
    student_w = student_model.layers[0].weight.detach().numpy()
    teacher_w = teacher_model.layers[0].weight.detach().numpy()
    # return the final loss
    return losses, student_w, teacher_w


