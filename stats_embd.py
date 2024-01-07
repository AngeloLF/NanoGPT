import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.optimize import curve_fit

"""

Ce fichier permet de comparer l'influence de l'embedding sur l'apprentissage

On a 8 modèles : 
- 4 avec 16 d'embd
- 4 avec 128 d'embd

Les 4 ont des batch / block size tel que : 
- 2 avec 16 de batch : avec 32 et 128 de block
- 2 avec 32 de batch : avec 32 et 128 de block

"""

folder = './saveStat/embd/'

dicoVal = []
xmin, xmax = 1, 1600

for file in os.listdir(folder):

	if '.stat' in file:

		l = None
		deco = file[:-5].split('_')

		if   '_embd16.stat' in file : l = 'r'
		elif '_embd128.stat' in file : l = 'b'
		else : l = 'g'

		if deco[0] == '16' and deco[1] == '32'  : 
			if   '_embd16.stat' in file :  color = '#000066'; l = '16-32 | e16'
			elif '_embd128.stat' in file : color = '#660000'; l = '16-32 | e128'
			else                : color = '#000000'

		elif deco[0] == '16' and deco[1] == '128' : 
			if   '_embd16.stat' in file :  color = '#000099'; l = '16-128 | e16'
			elif '_embd128.stat' in file : color = '#990000'; l = '16-128 | e128'
			else                : color = '#000000'

		elif deco[0] == '32' and deco[1] == '32'  : 
			if   '_embd16.stat' in file :  color = '#0000cc'; l = '32-32 | e16'
			elif '_embd128.stat' in file : color = '#cc0000'; l = '32-32 | e128'
			else                : color = '#000000'

		elif deco[0] == '32' and deco[1] == '128' : 
			if   '_embd16.stat' in file :  color = '#0000ff'; l = '32-128 | e16'
			elif '_embd128.stat' in file : color = '#ff0000'; l = '32-128 | e128'
			else                : color = '#000000'
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


plt.plot([], [], linestyle='-', color='b', label='Embeding de 16')
plt.plot([], [], linestyle='-', color='r', label='Embeding de 128')
plt.plot([], [], linestyle=':', color='k', label='loss value')
plt.xlabel('Temps (min)')
plt.ylabel('Loss')
plt.xlim([0.1, 5])
plt.ylim([1, 3])
# plt.xscale('log')
plt.legend()
plt.show()