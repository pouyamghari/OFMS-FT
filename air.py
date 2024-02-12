import numpy as np
import argparse
import time
import pandas as pd
from numpy import linalg as LA
import tensorflow as tf
from lib.datasets import data_processing
from lib.models import load_models
from lib.OFMSFT import OFMSFT_Client
from lib.OFMSFT import OFMSFT_Server

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='Air', type=str)
parser.add_argument("--num_samples", default=200, type=int)
parser.add_argument("--budget", default=5, type=int)
parser.add_argument("--communication_budget_fraction", default=.5, type=float)
parser.add_argument("--batch_size", default=50, type=int)


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

args = parser.parse_args()

num_samples = args.num_samples

C, X, Y = data_processing(args)
print('The number of clients: %s' %C)
    
M, N = X[0].shape
M*=C

exp, c = load_models(args)
K = len(exp)

B = args.budget
Omega = args.communication_budget_fraction*B*C
mem_batch = args.batch_size
eta = 10/np.sqrt(np.ceil(M/C))
T = num_samples
eta_f = 1e-3/np.sqrt(T)
mse_cl_ofmsft = np.zeros((C,T))
regret_ofmsft = np.zeros((C,T))
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
    
if args.dataset=="Air" or args.dataset=="WEC":
    task = "regression"
else:
    task = "classification"

ofmsft_server = OFMSFT_Server(task,c,Omega,exp,C)

ofmsft_clients = {}
for i in range(C):
    ofmsft_clients[i] = OFMSFT_Client(task,eta,eta_f,clusters)

for t in range(T):
    aggregated_models = {}
    I = []
    selected_models = {}
    bandwidth = np.zeros((C,1))
    for i in range(C):
        start_time = time.time()
        chosen_model, chosen_cluster = ofmsft_clients[i].modelselection()
        I.append(chosen_model)
        selected_models[i] = []
        for model in clusters[chosen_model][chosen_cluster]:
            selected_models[i].append(model)
        selected_models[i].append(chosen_model)
        bandwidth[i,0] = np.sum(c[selected_models[i],0])
        end_time = time.time()
        run_time_ofmsft[i,0]+=(end_time - start_time)
    if t>=mem_batch:
        selected_clients, alpha = ofmsft_server.Client_Selection(bandwidth)
    for i in range(C):
        start_time = time.time()
        L = np.zeros((1,K))
        for j in selected_models[i]:
            L[0,j] = exp[j].evaluate(X[i][t:t+1,:], Y[i][t:t+1,:])
        ofmsft_clients[i].updateclient(L)
        mse_cl_ofmsft[i,t] = exp[I[i]].evaluate(X[i][t:t+1,:], Y[i][t:t+1,:])
        if t>=mem_batch:
            if i in selected_clients:
                updating_models = []
                for ind in selected_models[i]:
                    updating_models.append(exp[ind])
                aggregated_models[i] = \
                ofmsft_clients[i].client_model_update(X[i][t-mem_batch:t+1,:], Y[i][t-mem_batch:t+1,:], 1, selected_models[i], updating_models, alpha)
        end_time = time.time()
        run_time_ofmsft[i,0]+=(end_time - start_time)
    if t>=mem_batch:
        ofmsft_server.server_model_update(aggregated_models)
        exp = ofmsft_server.model

print('MSE of OFMS-FT is %s' %np.mean(np.mean(mse_cl_ofmsft, axis=1)))
print('Standard deviation of OFMS-FT is %s' %np.std(np.mean(mse_cl_ofmsft, axis=1)))
print('Run time of OFMS-FT is %s' %np.mean(run_time_ofmsft,axis=0))