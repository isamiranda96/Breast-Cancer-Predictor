import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import csv

def training_graph(cost_vect, acc_vect, acc_vect_test):
    plt.plot(cost_vect, color='red', label = 'Costo Entrenamiento')
    plt.plot(acc_vect, color='blue', label = 'Accuracy Entrenamiento')
    plt.plot(acc_vect_test, color='orange', label = 'Accuracy Test')
    
    train_cost_patch = mpatches.Patch(color='red', label = 'Costo Entrenamiento')
    train_acc_patch = mpatches.Patch(color='blue', label = 'Accuracy Entrenamiento')
    test_acc_patch =mpatches.Patch(color='orange', label = 'Accuracy Test')
    plt.legend(handles=[train_cost_patch,train_acc_patch,test_acc_patch])
    plt.show()

def fitrar_nombre(x, feature_names, filtro):
    selecciones = np.isin(feature_names,filtro)
    return x[:,selecciones]

def load_log_book():
	try:
		return np.load('log_book.npy')
	except FileNotFoundError:
		return np.empty((0, 11))

def guardar_log_book(log_book):
	np.save('log_book.npy', log_book, allow_pickle=True)

def filter_log_book(log_book, nombre_modelo):
	modelo = log_book[log_book[:,0] == nombre_modelo,:]
	return  modelo[:,10][0]

def print_log_book(log_book):
	df = pd.DataFrame(columns=['Nombre Modelo','Numero de features','Numero de ejemplos','Alpha', \
		'Numero de Iteraciones','Accuracy Training','Accuracy Test', 'F1 score Test', 'Precision Test', 'Recall Test'])


	for experimento in log_book:
	    experimento_row = {'Nombre Modelo':experimento[0], 'Numero de features':experimento[1], \
	    'Numero de ejemplos':experimento[2], 'Alpha': experimento[3] , 'Numero de Iteraciones' : experimento[4], \
	    'Accuracy Training': experimento[5], 'Accuracy Test': experimento[6], 'F1 score Test': experimento[7], \
	    'Precision Test' : experimento[8], 'Recall Test' : experimento[9] }

	    df = df.append(experimento_row, ignore_index=True)

	return df


def get_normalizacion_vales(features):
	std = []
	media = []

	for i in list(range(0, features.shape[1])):
		std.append(np.std(features[:,i]))
		media.append(np.mean(features[:,i]))

	return std, media

        
