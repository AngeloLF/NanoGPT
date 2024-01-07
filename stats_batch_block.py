import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.optimize import curve_fit

"""

Ce fichier permet de comparer l'influence de la taille des block et batch sur l'apprentissage

On a deux cas :
- Influence du batch size : on fixe Batch a 16 et on fais varié block dans [32, 128, 256]
- Influence du block size : on fixe Block a 32 et on fais varié block dans [16, 32, 48]

"""

folder = './saveStat/batch_block/'

# Batch 16 : block variable

dicoVal = []
xmin, xmax = 1, 1600

for file in os.listdir(folder):

	if '.stat' in file:

		l = None
		deco = file[:-5].split('_')

		if deco[0] == '16':

			if deco[1] == '32'  : color = '#ff0000'; l = 'block size = 32'
			if deco[1] == '128' : color = '#00ff00'; l = 'block size = 128'
			if deco[1] == '256' : color = '#0000ff'; l = 'block size = 256'

		if l is not None:
			with open(folder + file, 'rb') as f:
				st = pickle.load(f) 
				dicoVal.append([None, None, st.timeTV_loss, st.train_loss, st.value_loss, color, l])
	


def funcTheo(x, x0, a, b, c):

	return 1 / np.log(x-x0) * a + (x-x0) * b + c


plt.figure(figsize=(8, 6))

for i, val in enumerate(dicoVal):

	xt, yt = val[2], val[3]
	xv, yv = val[2], val[4]

	stepX = np.arange(0, len(xt)) 

	popt, _ = curve_fit(funcTheo, xt, yt)
	popv, _ = curve_fit(funcTheo, xv, yv)
	X = np.linspace(xmin, xmax, 1000)


	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-', label=val[-1])
	plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':')


plt.xlabel('Temps (min)')
plt.ylabel('Loss')
plt.xlim([0.1, 25])
plt.ylim([1, 3])
plt.legend()
plt.show()







# Block 32 : batch variable

dicoVal = []
xmin, xmax = 1, 1600

for file in os.listdir(folder):

	if '.stat' in file:

		l = None
		deco = file[:-5].split('_')

		if deco[0] == '16':

			if deco[1] == '32'  : color = '#ff0000'; l = 'batch size = 16'

		if deco[0] == '32':

			if deco[1] == '32'  : color = '#00ff00'; l = 'batch size = 32'

		if deco[0] == '48':

			if deco[1] == '32'  : color = '#0000ff'; l = 'batch size = 48'

		if l is not None:
			with open(folder + file, 'rb') as f:
				st = pickle.load(f) 
				dicoVal.append([None, None, st.timeTV_loss, st.train_loss, st.value_loss, color, l])
	


def funcTheo(x, x0, a, b, c):

	return 1 / np.log(x-x0) * a + (x-x0) * b + c


plt.figure(figsize=(8, 6))

for i, val in enumerate(dicoVal):

	xt, yt = val[2], val[3]
	xv, yv = val[2], val[4]

	stepX = np.arange(0, len(xt)) 

	popt, _ = curve_fit(funcTheo, xt, yt)
	popv, _ = curve_fit(funcTheo, xv, yv)
	X = np.linspace(xmin, xmax, 1000)


	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-', label=val[-1])
	plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':')


plt.xlabel('Temps (min)')
plt.ylabel('Loss')
plt.xlim([0.1, 5])
plt.ylim([1, 3])
plt.xscale('log')
plt.legend()
plt.show()