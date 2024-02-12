import numpy as np
import argparse
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from lib.datasets import data_processing
from lib.models import load_models
from lib.OFMSFT import OFMSFT_Client
from lib.OFMSFT import OFMSFT_Server

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='CIFAR-10', type=str)
parser.add_argument("--budget", default=5, type=int)
parser.add_argument("--communication_budget_fraction", default=1.1, type=float)
parser.add_argument("--batch_size", default=50, type=int)

args = parser.parse_args()

def FFD(c, B):
    K = c.shape[0]
    sorted_c = np.sort(c, axis=0)[::-1]
    arg_c = np.argsort(c, axis=0)[::-1]
    clusters = {}
    clusters[0] = []
    for i, m in enumerate(sorted_c):
        j = 0
        while j < len(clusters) and m + np.sum(c[clusters[j],0]) > B:
            j+=1
        if j < len(clusters):
            clusters[j].append(arg_c[i,0])
        else:
            clusters[j] = []
            clusters[j].append(arg_c[i,0])
    return clusters


X, Y = data_processing(args)
M = X.shape[0]

exp, c = load_models(args)
K = len(exp)

C = 50
B = args.budget
Omega = args.communication_budget_fraction*B*C
mem_batch = args.batch_size
eta = 10/np.sqrt(np.ceil(M/C))
T = int(np.floor(M/C))
eta_f = 1e-3/np.sqrt(T)
acc_cl_ofmsft = np.zeros((C,int(np.floor(M/C))))
regret_ofmsft = np.zeros((C,int(np.floor(M/C))))
run_time_ofmsft = np.zeros((C,1))

clusters = {}
for model in range(K):
    if model==0:
        c_model = c[model+1:,:]
    elif model==K-1:
        c_model = c[:model,:]
    else: c_model = np.append(c[:model,:], c[model+1:,:], axis=0)
    cluster = FFD(c_model, B-c[model,0])
    for i in range(len(cluster)):
        for j, k in enumerate(cluster[i]):
            if k>=model:
                cluster[i][j] = k+1
    clusters[model] = cluster

ofmsft_server = OFMSFT_Server(c,Omega,exp,C)

ofmsft_clients = {}
for i in range(C):
    ofmsft_clients[i] = OFMSFT_Client(eta,eta_f,clusters)
        
X_u = {}
Y_u = {}
for i in range(C):
    X_u[i], Y_u[i] = X[i*T:i*T+1,:,:,:], Y[i*T:i*T+1,:]

for t in range(T):
    aggregated_models = {}
    I = []
    selected_models = {}
    for i in range(C):
        start_time = time.time()
        chosen_model, chosen_cluster = ofmsft_clients[i].modelselection()
        I.append(chosen_model)
        selected_models[i] = []
        for model in clusters[chosen_model][chosen_cluster]:
            selected_models[i].append(model)
        selected_models[i].append(chosen_model)
        end_time = time.time()
        run_time_ofmsft[i,0]+=(end_time - start_time)
    for i in range(C):
        start_time = time.time()
        L = np.zeros((1,K))
        for j in selected_models[i]:
            L[0,j], _ = exp[j].evaluate(X[t+i*int(M/C):t+i*int(M/C)+1,:,:,:], Y[t+i*int(M/C):t+i*int(M/C)+1,:], verbose=2)
        ofmsft_clients[i].updateclient(L)
        lss, acc = exp[I[i]].evaluate(X[t+i*int(M/C):t+i*int(M/C)+1,:,:,:], Y[t+i*int(M/C):t+i*int(M/C)+1,:], verbose=2)
        acc_cl_ofmsft[i,t] = acc
        if t>=mem_batch:
            X_u[i] = np.append(X_u[i][1:,:,:,:], X[t+i*int(M/C):t+i*int(M/C)+1,:,:,:], axis=0)
            Y_u[i] = np.append(Y_u[i][1:,:], Y[t+i*int(M/C):t+i*int(M/C)+1,:], axis=0)
        elif t>0 and t<mem_batch:
            X_u[i] = np.append(X_u[i], X[t+i*int(M/C):t+i*int(M/C)+1,:,:,:], axis=0)
            Y_u[i] = np.append(Y_u[i], Y[t+i*int(M/C):t+i*int(M/C)+1,:], axis=0)
        if t>=mem_batch:
            updating_models = []
            for ind in selected_models[i]:
                updating_models.append(exp[ind])
            aggregated_models[i] = \
            ofmsft_clients[i].client_model_update(X_u[i], Y_u[i], 1, selected_models[i], updating_models, 1)
        end_time = time.time()
        run_time_ofmsft[i,0]+=(end_time - start_time)
    if t>=mem_batch:
        ofmsft_server.server_model_update(aggregated_models)
        exp = ofmsft_server.model

print('Accuracy of OFMS-FT is %s' %np.mean(np.mean(acc_cl_ofmsft, axis=1)))
print('Standard deviation of OFMS-FT is %s' %np.std(np.mean(acc_cl_ofmsft, axis=1)))
print('Run time of OFMS-FT is %s' %np.mean(run_time_ofmsft,axis=0))