import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.optimize import curve_fit

"""

Ce fichier permet de comparer l'influence de la taille d'un texte sur l'apprentissage

On a :
- un texte de 200 milles tokens
- un texte de 1 million de tokens
- un texte de 7 millions de tokens

"""

folder = './saveStat/taille/'

dicoVal = []
xmin, xmax = 1, 1800

for file in os.listdir(folder):

	l = None

	if '7M.stat' in file:
		color = '#00cc00'
		l = '7M tokens'

	if '1M.stat' in file:
		color = '#ee0000'
		l = '1M tokens'

	if '200m.stat' in file:
		color = '#0000ee'
		l = '0.2M tokens'

	if l is not None:
		with open(folder + file, 'rb') as f:
			st = pickle.load(f)
			dicoVal.append([None, None, st.timeTV_loss, st.train_loss, st.value_loss, color, l])


def funcTheo(x, x0, a, b, c):

	return 1 / np.log(x-x0) * a + (x-x0) * b + c


plt.figure(figsize=(8, 6))

for val in dicoVal:
	xt, yt = val[2], val[3]
	xv, yv = val[2], val[4]

	stepX = np.arange(0, len(xt))

	popt, _ = curve_fit(funcTheo, xt, yt)
	popv, _ = curve_fit(funcTheo, xv, yv)
	X = np.linspace(xmin, xmax, 1000)

	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-', label=val[6])
	plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':') # loss value



plt.xlabel('Temps (en min)')
plt.ylabel('Loss')
plt.xlim([1, 30])
plt.ylim([1, 3])
plt.legend()
plt.show()