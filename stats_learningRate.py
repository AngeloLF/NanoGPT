import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.optimize import curve_fit

"""

Ce fichier permet de comparer l'influence du learning rate sur l'apprentissage

On a 12 modèles : 
- 4 avec 10^(-2) de learning rate
- 4 avec 10^(-3) de learning rate
- 4 avec 10^(-4) de learning rate

Les 4 ont des batch / block size tel que : 
- 2 avec 16 de batch : avec 32 et 128 de block
- 2 avec 32 de batch : avec 32 et 128 de block

"""

folder = './saveStat/learningRate/'

dicoVal = []
xmin, xmax = 1, 360

for file in os.listdir(folder):

	if '.stat' in file:

		l = None
		deco = file[:-5].split('_')

		if   '_l-2.stat' in file : color = 'r'
		elif '_l-3.stat' in file : color = 'g'
		elif '_l-4.stat' in file : color = 'b'
		else : l = 'k' # problemes

		if deco[0] == '16' and deco[1] == '32'  : 
			if   '_l-2.stat' in file : color = '#660000'; l = '-'
			elif '_l-3.stat' in file : color = '#006600'; l = '-'
			elif '_l-4.stat' in file : color = '#000066'; l = '-'
			else : l = 'k' # problemes

		elif deco[0] == '16' and deco[1] == '128' : 
			if   '_l-2.stat' in file : color = '#990000'; l = '-'
			elif '_l-3.stat' in file : color = '#009900'; l = '-'
			elif '_l-4.stat' in file : color = '#000099'; l = '-'
			else : l = 'k' # problemes

		elif deco[0] == '32' and deco[1] == '32'  : 
			if   '_l-2.stat' in file : color = '#cc0000'; l = '-'
			elif '_l-3.stat' in file : color = '#00cc00'; l = '-'
			elif '_l-4.stat' in file : color = '#0000cc'; l = '-'
			else : l = 'k' # problemes


		elif deco[0] == '32' and deco[1] == '128' : 
			if   '_l-2.stat' in file : color = '#ff0000'; l = '-'
			elif '_l-3.stat' in file : color = '#00ff00'; l = '-'
			elif '_l-4.stat' in file : color = '#0000ff'; l = '-'
			else : l = 'k' # problemes
		else:
			print(deco) # pour debug

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


	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-')
	plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':')


plt.plot([], [], linestyle='-', color='r', label='Learning Rate = 1e-2')
plt.plot([], [], linestyle='-', color='g', label='Learning Rate = 1e-3')
plt.plot([], [], linestyle='-', color='b', label='Learning Rate = 1e-4')
plt.plot([], [], linestyle=':', color='k', label='loss value')
plt.xlabel('Temps (min)')
plt.ylabel('Loss')
plt.xlim([0.15, 5])
plt.ylim([1.40, 4])
# plt.xscale('log')
plt.legend()
plt.show()