import pandas as pd
import numpy as np

def data_processing(args):
    if args.dataset=="Air":
        num_samples = args.num_samples
        data_Aotizhongxin = pd.read_csv('Air/PRSA_Data_Aotizhongxin.csv', delimiter=',')
        data_Aotizhongxin = data_Aotizhongxin.dropna()
        wind_direction = data_Aotizhongxin.wd.unique()
        for i in range(len(wind_direction)):
            data_Aotizhongxin = data_Aotizhongxin.replace(wind_direction[i], i)
        data_Aotizhongxin = data_Aotizhongxin.to_numpy()
        data_Aotizhongxin = data_Aotizhongxin[:,2:-1]
        data_Aotizhongxin = data_Aotizhongxin[:10000,:].astype('float32')

        data_Changping = pd.read_csv('Air/PRSA_Data_Changping.csv', delimiter=',')
        data_Changping = data_Changping.dropna()
        wind_direction = data_Changping.wd.unique()
        for i in range(len(wind_direction)):
            data_Changping = data_Changping.replace(wind_direction[i], i)
        data_Changping = data_Changping.to_numpy()
        data_Changping = data_Changping[:,2:-1]
        data_Changping = data_Changping[:10000,:].astype('float32')

        K_1, K_2 = int(data_Aotizhongxin.shape[0]/num_samples), int(data_Changping.shape[0]/num_samples)
        C = K_1 + K_2

        X = {}
        Y = {}
        for i in range(K_1):
            client_data = data_Aotizhongxin[200*i:200*(i+1),:]
            X[i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
            Y[i] = client_data[:,6:7]
        for i in range(K_2):
            client_data = data_Changping[200*i:200*(i+1),:]
            X[K_1 + i] = np.concatenate((client_data[:,:6],client_data[:,7:]),axis=1)
            Y[K_1 + i] = client_data[:,6:7]

        X_norm_max_clients = np.zeros((1,C))
        Y_max_clients = np.zeros((1,C))
        Y_min_clients = np.zeros((1,C))
        for i in range(C):
            X_n = np.zeros((1,num_samples))
            for j in range(num_samples):
                X_n[0,j] = LA.norm(X[i][j,:])
            X_norm_max_clients[0,i] = np.max(X_n)
            Y_max_clients[0,i] = np.max(Y[i])
            Y_min_clients[0,i] = np.min(Y[i])
        X_norm_max = np.max(X_norm_max_clients)
        Y_max = np.max(Y_max_clients)
        Y_min = np.min(Y_min_clients)

        for i in range(C):
            for j in range(num_samples):
                X[i][j,:] = X[i][j,:]/X_norm_max
            Y[i] = (Y[i] - Y_min)/(Y_max - Y_min)
        
        return C, X, Y
    elif args.dataset=="WEC":
        num_samples = args.num_samples
        data_a = pd.read_csv('Adelaide_Data.csv', delimiter=',')
        data_a = data_a.dropna()
        data_a = data_a.to_numpy()
        data_a = data_a[:10000,:]

        data_p = pd.read_csv('Perth_Data.csv', delimiter=',')
        data_p = data_p.dropna()
        data_p = data_p.to_numpy()
        data_p = data_p[:10000,:]

        K_1, K_2 = int(data_a.shape[0]/num_samples), int(data_p.shape[0]/num_samples)
        C = K_1 + K_2

        X = {}
        Y = {}
        for i in range(K_1):
            client_data = data_a[200*i:200*(i+1),:]
            X[i] = client_data[:,:-1]
            Y[i] = client_data[:,-1:]
        for i in range(K_2):
            client_data = data_p[200*i:200*(i+1),:]
            X[K_1 + i] = client_data[:,:-1]
            Y[K_1 + i] = client_data[:,-1:]

        X_norm_max_clients = np.zeros((1,C))
        Y_max_clients = np.zeros((1,C))
        Y_min_clients = np.zeros((1,C))
        for i in range(C):
            X_n = np.zeros((1,num_samples))
            for j in range(num_samples):
                X_n[0,j] = LA.norm(X[i][j,:])
            X_norm_max_clients[0,i] = np.max(X_n)
            Y_max_clients[0,i] = np.max(Y[i])
            Y_min_clients[0,i] = np.min(Y[i])
        X_norm_max = np.max(X_norm_max_clients)
        Y_max = np.max(Y_max_clients)
        Y_min = np.min(Y_min_clients)

        for i in range(C):
            for j in range(num_samples):
                X[i][j,:] = X[i][j,:]/X_norm_max
            Y[i] = (Y[i] - Y_min)/(Y_max - Y_min)
            
        return C, X, Y
    elif args.dataset=="CIFAR-10":
        X = np.load('CIFAR-10/test_images_cifar_10.npy')
        Y = np.load('CIFAR-10/test_labels_cifar_10.npy')
        
        return X, Y
    elif args.dataset=="MNIST":
        X = np.load('MNIST/test_images_mnist.npy')
        Y = np.load('MNIST/test_labels_mnist.npy')
        
        return X, Y
    
data_processing(args)