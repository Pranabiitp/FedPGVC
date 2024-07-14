


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)






def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict






def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss





def calculate_fedprox_regularization(global_model, local_model, mu):
    regularization_term = 0.0
    global_weights = global_model.get_weights()
    local_weights = local_model.get_weights()

    for global_w, local_w in zip(global_weights, local_weights):
        regularization_term += 0.5 * mu * tf.reduce_sum(tf.square(global_w - local_w))

    return regularization_term





def avg_weights(scaled_weight_list):
    '''Return the average of the listed scaled weights.'''
    num_clients = len(scaled_weight_list)
    
    if num_clients == 0:
        return None  # Handle the case where the list is empty
        
    avg_grad = list()
    
    # Get the sum of gradients across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / num_clients
        avg_grad.append(layer_mean)
        
    return avg_grad





import tensorflow as tf

# Load CIFAR-100 dataset
(_, _), (test, test_labels) = tf.keras.datasets.cifar100.load_data()

# Convert test labels to one-hot encoded format
num_classes = 100  # CIFAR-100 has 100 classes
test_labels_one_hot = tf.one_hot(test_labels, num_classes)

# Print the shape of the one-hot encoded labels
print("Shape of one-hot encoded test labels:", test_labels_one_hot.shape)

# Convert the one-hot encoded labels to numpy array and remove the extra dimension
one_hot_labels = test_labels_one_hot.numpy()
label = one_hot_labels.squeeze(axis=1)

# Print the updated shape of the labels
print("Updated shape of labels:", label.shape)





for i in range(1, 11):
    globals()[f"train{i}"] = globals()[f"train{i}"] / 255



test=test/255





client_data1 = {
#     'client0': (test, label),
    'client1': (test, label),
    'client2': (test, label),
    'client3': (test, label),
    'client4': (test, label),
    'client5': (test, label),
    'client6': (test, label),
    'client7': (test, label),
    'client8': (test, label),
    'client9': (test, label),
    'client10': (test, label)


#     'client6': (test, label)
    
}
#create clients
test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    'client5': (train5, label5),
    'client6': (train6, label6),
    'client7': (train7, label7),
    'client8': (train8, label8),
    'client9': (train9, label9),
    'client10': (train10, label10)

    
}
#create clients
clients_batched = create_clients(client_data2)





client_names = list(clients_batched.keys())






# Your WDRO variables
eta_g = 1  # Global learning rate
eta_l = 0.02 # Local learning rate
R = 100  # Number of communication rounds
p = [0, 0,0,0,0,0,0, 0, 1]  # Mask: 1 for last two layers, 0 for others
S_sgd = [i for i, val in enumerate(p) if val == 0]
S_wdro = [i for i, val in enumerate(p) if val == 1]
mu=0.2
# Number of clients and local epochs
N = len(clients_batched)
client_epochs = {'client1': 1, 'client2': 1, 'client3': 1, 'client4': 1}  # Adjust as needed

comms_round = 100  # Number of global epochs
acc3 = []
loss3 = []
train_acc_clients = [[] for _ in range(N)]  # List of lists for training accuracy for each client
val_acc_clients = [[] for _ in range(N)]    # List of lists for validation accuracy for each client
best_acc = 0
best_weights = None

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    local_weight_list = []

    # Randomize client data - using keys
    client_names = list(clients_batched.keys())

    for i, client in enumerate(client_names):
#         smlp_global = SimpleMLP()
        local_model = create_cnn_model()
        local_model.set_weights(global_weights)

        history = local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),
            validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
            epochs=1,
            batch_size=16,
            verbose=2
        )
        local_weights = local_model.get_weights()

            # Apply FedProx regularization
#             fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)
        for i in range(len(local_weights)):
                local_weights[i] += mu * (global_weights[i] - local_weights[i])
        # Calculate gradients
        with tf.GradientTape() as tape:
            y_pred = local_model(np.array(clients_batched[client][0]))
            batch_loss = tf.keras.losses.categorical_crossentropy(np.array(clients_batched[client][1]), y_pred)

        # Compute gradients
        batch_grad = tape.gradient(batch_loss, local_model.trainable_weights)

        # Calculate rho_i
        rho_i = [np.mean(b_loss * b_grad) for b_loss, b_grad in zip(batch_loss, batch_grad)]

        # Update local model weights
        for j, layer in enumerate(local_model.layers):
            if j in S_sgd:
#                 print("SGD: Layer index =", j)
                updated_weights = layer.get_weights()
            elif j in S_wdro:
#                 print("WDRO: Layer index =", j)
                rho_j = rho_i[j]  # Get the corresponding rho_i for this layer
                updated_weights = [w - eta_l * rho_j * w_delta for w, w_delta in zip(layer.get_weights(), layer.get_weights())]
            else:
                raise ValueError("Unexpected layer index")

            # Set the updated weights to the layer
            layer.set_weights(updated_weights)

        weights = local_model.get_weights()
        local_weight_list.append(weights)
#         K.clear_session()

    # Calculate the average weights across all clients for each layer
    average_weights = avg_weights(local_weight_list)

    # Update the global model with the average weights
    global_model.set_weights(average_weights)

    # Test the global model and print out metrics after each communications round
    global_acc, global_loss = test_model(test, label, global_model, comm_round)
    acc3.append(global_acc)
    loss3.append(global_loss)

    if global_acc > best_acc:
        best_acc = global_acc
        best_weights = global_model.get_weights()



# Set the global model to the best weights found during training
global_model.set_weights(best_weights)







