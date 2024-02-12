import numpy as np
import tensorflow as tf
from tensorflow import keras
    
class OFMSFT_Client:
    def __init__(self,task,eta,eta_f,clusters):
        self.eta = eta
        self.eta_f = eta_f
        self.clusters = clusters
        self.num_models = len(clusters)
        self.z = np.ones((len(clusters),1))
        self.p = self.z/np.sum(self.z)
        self.task = task
    
    def modelselection(self):
        self.p = self.z/np.sum(self.z)
        I = 0
        r = np.random.rand()
        while r > np.sum(self.p[:I+1,0]):
            I+=1
        r = np.random.rand()
        J = 0
        while r>(J+1)/len(self.clusters[I]):
            J+=1
        return I, J
    
    def updateclient(self, L):
        if self.task=="regression":
            q = np.zeros((self.num_models,1))
            for i in range(self.num_models):
                q[i,0] = ( 1 - (1/len(self.clusters[i])) )*self.p[i,0]
                for j in range(self.num_models):
                    q[i,0]+=self.p[j,0]/len(self.clusters[j])
            for i in range(self.num_models):
                self.z[i,0]*=np.exp(-(self.eta*L[0,i])/q[i,0])
            self.q = q
        elif self.task=="classification":
            q = np.zeros((self.num_models,1))
            for i in range(self.num_models):
                q[i,0] = ( 1 - (1/len(self.clusters[i])) )*self.p[i,0]
                for j in range(self.num_models):
                    q[i,0]+=self.p[j,0]/len(self.clusters[j])
            for i in range(self.num_models):
                self.z[i,0]*=np.exp(-(self.eta*L[0,i])/q[i,0])
            self.q = q
            
    def client_model_update(self, X_batch, Y_batch, num_epochs, selected_models, models, alpha):
        if self.task=="regression":
            local_model = {}
            client_batch_size = np.prod(Y_batch.shape)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_batch, Y_batch))
            loss_fn = keras.losses.MeanSquaredError()
            for j, i in enumerate(selected_models):
                local_model[i] = models[j]
                if self.q[i,0]>.05:
                    optimizer = keras.optimizers.SGD(learning_rate=(alpha*self.eta_f/self.q[i,0]))
                else: optimizer = keras.optimizers.SGD(learning_rate=(alpha*20*self.eta_f))
                for epoch in range(num_epochs):
                    with tf.GradientTape() as tape:
                        logits = local_model[i](X_batch, training=True)
                        loss_value = loss_fn(Y_batch, logits)
                    grads = tape.gradient(loss_value, local_model[i].trainable_weights)
                    optimizer.apply_gradients(zip(grads, local_model[i].trainable_weights))
            return local_model
        elif self.task=="classification":
            local_model = {}
            client_batch_size = np.prod(Y_batch.shape)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_batch, Y_batch))
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            for j, i in enumerate(selected_models):
                local_model[i] = models[j]
                if self.q[i,0]>.05:
                    optimizer = keras.optimizers.SGD(learning_rate=(alpha*self.eta_f/self.q[i,0]))
                else: optimizer = keras.optimizers.SGD(learning_rate=(alpha*20*self.eta_f))
                for epoch in range(num_epochs):
                    with tf.GradientTape() as tape:
                        logits = local_model[i](X_batch, training=True)
                        loss_value = loss_fn(Y_batch, logits)
                    grads = tape.gradient(loss_value, local_model[i].trainable_weights)
                    optimizer.apply_gradients(zip(grads, local_model[i].trainable_weights))
            return local_model
        
class OFMSFT_Server:
    def __init__(self,task,b,Omega,models,num_clients):
        self.b = b
        self.num_clients = num_clients
        self.Omega = Omega
        self.model = models
        self.task = task
                    
    def Client_Selection(self, bandwidth):
        self.client_clusters = {}
        self.client_clusters[0] = []
        sorted_bandwidth = np.sort(bandwidth, axis=0)[::-1]
        arg_bandwidth = np.argsort(bandwidth, axis=0)[::-1]
        for i, m in enumerate(sorted_bandwidth):
            j = 0
            while j < len(self.client_clusters) and m + np.sum(bandwidth[self.client_clusters[j],0]) > self.Omega:
                j+=1
            if j < len(self.client_clusters):
                self.client_clusters[j].append(arg_bandwidth[i,0])
            else:
                self.client_clusters[j] = [arg_bandwidth[i,0]]
        r = np.random.rand()
        i = 0
        while r>(i+1)/len(self.client_clusters):
            i+=1
        self.selected_client_cluster = i
        return self.client_clusters[i], len(self.client_clusters)
    
    def server_model_update(self, aggregated_models):
        if self.task=="regression":
            local_models = {}
            for i in range(len(self.model)):
                local_models[i] = []
            for client in self.client_clusters[self.selected_client_cluster]:
                for model_index in range(len(self.model)):
                    if model_index in aggregated_models[client].keys():
                        local_models[model_index].append(aggregated_models[client][model_index])
                    else:
                        local_models[model_index].append(self.model[model_index])
            for i in range(len(self.model)):
                if len(local_models[i])>0:
                    global_weights = []
                    for layer in range(len(local_models[i][0].get_weights())):
                        layer_weights = np.array([local_model.get_weights()[layer] for local_model in local_models[i]])
                        avg_layer_weights = np.mean(layer_weights, axis=0)
                        global_weights.append(avg_layer_weights)
                    self.model[i].set_weights(global_weights)
        elif self.task=="classification":
            local_models = {}
            for i in range(len(self.model)):
                local_models[i] = []
            for client in self.client_clusters[self.selected_client_cluster]:
                for model_index in range(len(self.model)):
                    if model_index in aggregated_models[client].keys():
                        local_models[model_index].append(aggregated_models[client][model_index])
                    else:
                        local_models[model_index].append(self.model[model_index])
            for i in range(len(self.model)):
                if len(local_models[i])>0:
                    global_weights = []
                    for layer in range(len(local_models[i][0].get_weights())):
                        layer_weights = np.array([local_model.get_weights()[layer] for local_model in local_models[i]])
                        avg_layer_weights = np.mean(layer_weights, axis=0)
                        global_weights.append(avg_layer_weights)
                    self.model[i].set_weights(global_weights)