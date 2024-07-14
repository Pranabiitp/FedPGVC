## Harnessing Heterogeneity: Improving Partial Variance Control in Federated Learning for Image Classification
This repository contains the code for the paper "Harnessing Heterogeneity: Improving Partial Variance Control in Federated Learning for Image Classification".

# Dependencies
- Tensorflow = 2.10.0
- scikit-learn = 1.3.2

# Data Preparing
To divide the dataset into the aprequired no. of clients, run Data Prepration.py and choose the required dataset (CIFAR100, MNIST or FMNIST) and then change the degree of heterogenity (alpha) as required. you will get the desired distribution for each client.

# Model Structure
To choose the appropriate  model, run Models.py and choose the required model for each of the dataset.

# Run FedPGVC
After done with above process, you can run the FedPMVR, our proposed method.

# Run FedPGVC + Existing algorithms
To incorporate FedPGVC with FedProx, FedNova, and FedBN run corresponding file e.g. FedProx + FedPGVC.py. 

# Evaluation
After federated training, run Evaluation.py to acess the evaluation metrics such as accuracy, precision, recall etc.

