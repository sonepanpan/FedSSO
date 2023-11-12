import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from keras.datasets import cifar10, mnist
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.backend import image_data_format
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
import copy

from build_model import Model, get_training_model
import csv
import random
import time

method_name = "FedSSO_new_apt"

# dataset_name = 'eminst'
dataset_name = 'minst'
# dataset_name = 'cifar10'

# client config
NUMOFCLIENTS = 10 # number of client(as particles)
SELECT_CLIENTS = 0.5 # c
ROUND = 30 # number of total iteration
CLIENT_EPOCHS = 5 # number of each client's iteration
BATCH_SIZE = 10 # Size of batches to train on

DROP_RATE = 0 # 0 ~ 1.0 float value

SSO_para = [0.3, 0.5, 0.7]

# model config 
LOSS = 'categorical_crossentropy' # Loss function
NUMOFCLASSES = 10 # Number of classes
lr = 0.0025
OPTIMIZER = SGD(learning_rate=lr, momentum=0.9, decay=lr/(ROUND*CLIENT_EPOCHS), nesterov=False) # lr = 0.015, 67 ~ 69%


def write_csv(method_name, dataset_name, list):
    """
    Make the result a csv file.

    Args: 
        algorithm_name: algorithm name, string type ex) FedPSO, FedAvg
        list: accuracy list, list type
    """
    file_name = f'{method_name}_{dataset_name}_Cg_{SSO_para[0]}_Cp_{SSO_para[1]}_Cw_{SSO_para[2]}_randomDrop_{DROP_RATE}%_output_LR_{lr}_CLI_{NUMOFCLIENTS}_CLI_EPOCHS_{CLIENT_EPOCHS}_TOTAL_EPOCHS_{ROUND}_BATCH_{BATCH_SIZE}.csv'
    #file_name = file_name.format(drop=DROP_RATE, name=algorithm_name, lr=lr, cli=NUMOFCLIENTS, cli_epoch=CLIENT_EPOCHS, epochs=EPOCHS, batch=BATCH_SIZE)
    f = open(file_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()


def load_dataset(dataset_name):
    """
    This function loads the dataset provided by Keras and pre-processes it in a form that is good to use for learning.
    
    Return:
        (X_train, Y_train), (X_test, Y_test)
    """

    if dataset_name == 'cifar10':
        # Code for experimenting with CIFAR-10 datasets.
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    if dataset_name == 'minst':
        # Code for experimenting with MNIST datasets.
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)



def init_model(x_train_shape, y_train_shape):
    model = get_training_model(x_train_shape, y_train_shape)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["accuracy"])

    return model


def client_data_config(x_train, y_train):
    """
    Split the data set to each client. Split randomly selected data from the total dataset by the same size.
    
    Args:
        x_train: Image data to be used for learning
        y_train: Label data to be used for learning

    Returns:
        A dataset consisting of a list type of client sizes
    """
    client_data = [() for _ in range(NUMOFCLIENTS)] # () for _ in range(NUMOFCLIENTS)
    num_of_each_dataset = int(x_train.shape[0] / NUMOFCLIENTS)

    for i in range(NUMOFCLIENTS):
        split_data_index = []
        while len(split_data_index) < num_of_each_dataset:
            item = random.choice(range(x_train.shape[0]))
            if item not in split_data_index:
                split_data_index.append(item)
        
        new_x_train = np.asarray([x_train[k] for k in split_data_index])
        new_y_train = np.asarray([y_train[k] for k in split_data_index])

        client_data[i] = (new_x_train, new_y_train)

    return client_data

class particle():
    def __init__(self, particle_num, client, x_train, y_train):
        # for check particle id
        self.particle_id = particle_num
        
        # particle model init
        self.particle_model = client
        
        # best model init
        self.local_best_model = client
        self.global_best_model = client

        # best score init
        self.local_best_score = 0.0
        self.global_best_score = 0.0

        # x & y
        self.x = x_train
        self.y = y_train
        
        #SSO parameter setting
        self.Cg=SSO_para[0]
        self.Cp=SSO_para[1]
        self.Cw=SSO_para[2]

        # local gradient
        self.local_gradient=np.array(client.get_weights(), dtype=object)

    def train_particle(self):
        print(f"client {self.particle_id+1}/{NUMOFCLIENTS} fitting")

        # set each epoch's weight
        step_model = self.particle_model
        step_weight = step_model.get_weights()
        
        # new_velocities = [None] * len(step_weight)
        new_weight = [None] * len(step_weight)
        rnd = random.random()
        beta = random.random()
        print(f"beta: {beta}")

        # SSO algorithm applied to weights
        # for index, layer in enumerate(step_weight):
        #     if   0<=rnd and rnd<self.Cg:
        #         sol_name = 'current weight'
        #         new_weight[index] = step_weight[index]
        #     elif self.Cg<=rnd and rnd<self.Cp: 
        #         sol_name = 'globel best weight'
        #         new_weight[index] = self.global_best_model.get_weights()[index]
        #     elif self.Cp<=rnd and rnd<=1:
        #         sol_name = 'personal best weight'
        #         new_weight[index] = self.local_best_model.get_weights()[index]

        
        
        for index, layer in enumerate(step_weight):
            velocities = np.random.rand(*layer.shape) / 5 - 0.10
            print(f"index-layer: {self.global_best_model.get_weights()[index] - layer}")
            if   0<=rnd and rnd<self.Cg:
                sol_name = 'current weight + global best'
                new_weight[index] = step_weight[index] + beta*velocities + beta*(self.global_best_model.get_weights()[index] - layer)
            elif self.Cg<=rnd and rnd<self.Cp: 
                sol_name = 'current weight + personal best'
                new_weight[index] = step_weight[index] + beta*velocities + beta*(self.local_best_model.get_weights()[index] - layer)
            elif self.Cp<=rnd and rnd<self.Cw:
                sol_name = 'global + personal best'
                new_weight[index] = step_weight[index] + beta*(self.global_best_model.get_weights()[index] - layer) + (1-beta)*(self.local_best_model.get_weights()[index]-layer)
            else:
                sol_name='random'
                new_weight[index] = step_weight[index] + beta*velocities + beta*(self.global_best_model.get_weights()[index] - layer) + (1-beta)*(self.local_best_model.get_weights()[index]-layer)
            # sol_name='random'
            # velocities = np.random.rand(*layer.shape) / 5 - 0.10
            # new_weight[index] = step_weight[index] + beta*velocities + beta*(self.global_best_model.get_weights()[index] - layer) + (1-beta)*(self.local_best_model.get_weights()[index]-layer)

        step_model.set_weights(new_weight)
        

        save_model_path = f'checkpoint_{method_name}/checkpoint_particle_{self.particle_id}'
        mc = ModelCheckpoint(filepath=save_model_path, 
                            monitor='val_loss', 
                            mode='min',
                            save_best_only=True,
                            save_weights_only=True,
                            )
        print(f"client {self.particle_id+1}/{NUMOFCLIENTS} fitting : " + sol_name)
        hist = step_model.fit(x=self.x, y=self.y,
                epochs=CLIENT_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                validation_split=0.2,
                callbacks=[mc],
                )

        train_score_acc = hist.history['val_accuracy'][-1]
        # train_score_loss = hist.history['val_loss'][-1]

        # calculate local gradient
        new_new_weight = step_model.get_weights()
        self.local_gradient = (np.array(new_weight, dtype=object) - np.array(new_new_weight, dtype=object)) / lr
        self.local_gradient = fully_flatten(self.local_gradient)
        # print(self.local_gradient.shape, np.transpose(self.local_gradient).shape)
        # print(np.inner(self.local_gradient, self.local_gradient))

        # print(new_new_weight)

        step_model.load_weights(save_model_path)

        self.particle_model = step_model

        #更新pBest
        if self.local_best_score <= train_score_acc:
            self.local_best_model = step_model
            print("---renew pBest")

        #更新gBest
        if self.global_best_score <= train_score_acc:
        # if self.global_best_score >= train_score_loss:
            self.global_best_model = step_model
            print("---renew gBest")
            
        return self.particle_id, train_score_acc, step_model
    
    def update_global_model(self, global_best_model, global_best_score):
        if self.local_best_score < global_best_score:    
            self.global_best_model = global_best_model
            self.global_best_score = global_best_score

    def resp_model(self, gid):
        if self.particle_id == gid:
            return self.particle_model



def get_best_score_by_loss(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 100000
    temp_index = 0


    for index, result in enumerate(step_result):
        if temp_score > result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]

def fully_flatten(array):
  array = np.squeeze(array)
  ans = np.squeeze(array[0])
  for i in range(1, len(array)):
    np.append(ans, array[i].flatten())
  ans = ans.reshape(1, -1)
  print(f'weight length: {ans.shape}')
  return ans


def get_best_score_by_acc(step_result):
    # step_result = [[step_model, train_socre_acc],...]
    temp_score = 0
    temp_index = 0

    for index, result in enumerate(step_result):
        if temp_score < result[1]:
            temp_score = result[1]
            temp_index = index

    return step_result[temp_index][0], step_result[temp_index][1]


def aggregation_with_para_c(server_result, sso_model):
    # server_result [[pid, score], ...]

    server_result.sort(key=lambda x: x[1], reverse=True)
    server_result_sorted = server_result[:round(SELECT_CLIENTS*NUMOFCLIENTS)]

    avg_weight = np.array(sso_model[server_result_sorted[0][0]].local_best_model.get_weights())
    
    if len(server_result_sorted) > 1:
        for i in range(1, len(server_result_sorted)):
            avg_weight += sso_model[server_result_sorted[i][0]].local_best_model.get_weights()
    
    avg_weight = avg_weight / round(SELECT_CLIENTS*NUMOFCLIENTS)

    return avg_weight

def aggregation_with_weight(weight, sso_model):
    avg_weight = np.array(sso_model[0].particle_model.get_weights(), dtype = object) * weight[0]
    if len(sso_model) > 1:
        for i in range(1, len(sso_model)):
            avg_weight += np.array(sso_model[i].particle_model.get_weights(), dtype = object) * weight[i]

    return avg_weight


def calculate_angle(W1, W2):
    # Reshape matrices to vectors
    vec1 = W1.flatten()
    vec2 = W2.flatten()

    # Compute dot product
    inner_product = np.inner(vec1, vec2)

    # Compute norms
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Calculate cosine of the angle
    cosine_theta = inner_product / (norm_vec1 * norm_vec2)
    
    # Ensure the value lies between -1 and 1 to avoid numerical errors
    cosine_theta = np.clip(cosine_theta, -1.0, 1.0)

    # Calculate the angle and return
    return np.arccos(cosine_theta)


def gompertz_model(angle, alpha=5) :
    return alpha*(1-np.exp(-1*np.exp(-1*alpha*(angle-1))))

def node_contribution(sso_model, smoothed_angle, r):
    global_gradient = sso_model[0].local_gradient
    for i, client in enumerate(sso_model):
        if i != 0:
            # print(f"particle {i}'s gradient: {np.shape(sso_model[i].local_gradient)}")
            global_gradient = np.add(sso_model[i].local_gradient, global_gradient)
    # print(global_gradient.shape)

    # print(f"testinggggg: {calculate_angle(global_gradient, global_gradient)}")
    for i, client in enumerate(sso_model):
        if r==0:
            smoothed_angle.append(calculate_angle(sso_model[i].local_gradient, global_gradient))
        else:
            smoothed_angle[i] = (smoothed_angle[i]*r+calculate_angle(sso_model[i].local_gradient, global_gradient))/(r+1)
    # print(f"smoothed_angle: {smoothed_angle}")

    node_cont=[]
    whole_node_cont = 0
    for i in range(len(smoothed_angle)):
        node_cont.append(np.exp(gompertz_model(smoothed_angle[i])))
        whole_node_cont += node_cont[i]
    # print(node_cont)
    # print(whole_node_cont)

    weight = []
    for i in range(len(node_cont)):
        weight.append(node_cont[i]/whole_node_cont)
    print(sum(weight))
  

    return sso_model, smoothed_angle, weight



if __name__ == "__main__":
    
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)

    start = time.time()
    #模型初始化
    server_model = init_model(x_train.shape[1:], NUMOFCLASSES)
    # server_model.summary()

    client_data = client_data_config(x_train, y_train)
    # print(client_data[0])

    #存每個particle(client)的model
    
    sso_model = []
    smoothed_angle = []
    for i in range(NUMOFCLIENTS):
        sso_model.append(particle(particle_num=i, client=init_model(client_data[i][0].shape[1:], NUMOFCLASSES), x_train=client_data[i][0], y_train=client_data[i][1]))

    server_evaluate_acc = []
    global_best_model = None
    global_best_score = 0.0

    for r in range(ROUND):
        server_result = []
        # start = time.time()

        # trianing...
        for client in sso_model:
            if r != 0:
                #更新上一round global score的資訊
                client.update_global_model(server_model, acc)
            
            # local_model, train_score = client.train_particle()
            # server_result.append([local_model, train_score])

            
            pid, train_score, train_model = client.train_particle()
            server_result.append([pid, train_score, ])
            
            # rand = random.randint(0, 99)

            # # Randomly dropped data sent to the server
            # drop_communication = range(DROP_RATE)
            # if rand not in drop_communication:
            #     server_result.append([pid, train_score])
            

        # node contribution
        sso_model, smoothed_angle, weight = node_contribution(sso_model, smoothed_angle, r)

        # model aggregation
        # Send the optimal model to each client after the best score comparison

        #average weight
        avg_weight_1 = aggregation_with_weight(weight, sso_model)
        avg_weight_2 = aggregation_with_para_c(server_result, sso_model)
        server_model.set_weights(avg_weight_1)
        acc_1 = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)[1]
        server_model.set_weights(avg_weight_2)
        acc_2 = server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)[1]

        if acc_1 > acc_2:
          print('node contribution useful')
          acc_2=acc_1
          server_model.set_weights(avg_weight_1)

        # global best weight
        gid, this_best_score = get_best_score_by_acc(server_result)
        for client in sso_model:
            if client.resp_model(gid) != None:
                this_best_model = client.resp_model(gid)
        acc_3=this_best_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)[1]
        if acc_2 < acc_3:
            print('global useful')
            server_model = this_best_model
            acc = acc_3
        else:
            acc=acc_2

        if global_best_score < acc:
            global_best_model=server_model
            global_best_score=acc

        print("server {}/{} evaluate".format(r+1, ROUND))
        server_evaluate_acc.append(server_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1))
    end = time.time()
    print(f"global model:{global_best_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)}")
    print(end-start)
    write_csv(method_name, dataset_name ,server_evaluate_acc)
