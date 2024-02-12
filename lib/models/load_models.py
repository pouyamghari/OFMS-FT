import tensorflow as tf
import numpy as np

def load_models(args):
    exp = []
    if args.dataset=="Air":
        for i in range(2):
            for j in range(5):
                model_name = 'Air/ff_air_' + str(i+1) + str(j+1) + '_d1.h5'
                exp.append(tf.keras.models.load_model(model_name))
        for i in range(2):
            for j in range(5):
                model_name = 'Air/ff_air_' + str(i+1) + str(j+1) + '_d2.h5'
                exp.append(tf.keras.models.load_model(model_name))
                
        costs = np.ones((len(exp),1))
        
        return exp, costs
    elif args.dataset=="WEC":
        for i in range(2):
            for j in range(5):
                model_name = 'WEC/ff_wec_' + str(i+1) + str(j+1) + '_d1.h5'
                exp.append(tf.keras.models.load_model(model_name))
        for i in range(2):
            for j in range(5):
                model_name = 'WEC/ff_wec_' + str(i+1) + str(j+1) + '_d2.h5'
                exp.append(tf.keras.models.load_model(model_name))
                
        costs = np.ones((len(exp),1))
        
        return exp, costs
    elif args.dataset=="CIFAR-10":
        c = np.zeros((K,1))
        exp = []

        for i in range(10):
            filename = 'CIFAR-10/vgg_2_' + str(i) + '_cifar.h5'
            model = load_model(filename)
            exp.append(model)
            filename = 'CIFAR-10/vgg_3_' + str(i) + '_cifar.h5'
            model = load_model(filename)
            exp.append(model)

        for i in range(0,K):
            if i%2==0:
                c[i,0] = 273066/307114
            else: c[i,0] = 1
        
        return exp, c
    elif args.dataset=="MNIST":
        c = np.zeros((K,1))
        exp = []

        for i in range(10):
            filename = 'MNIST/vgg_1_' + str(i) + '_mnist.h5'
            model = load_model(filename)
            exp.append(model)
            filename = 'MNIST/vgg_2_' + str(i) + '_mnist.h5'
            model = load_model(filename)
            exp.append(model)

        for i in range(K):
            if i%2==1:
                c[i,0] = 159254/240528
            else: c[i,0] = 1
        
        return exp, c
    
load_models(args)