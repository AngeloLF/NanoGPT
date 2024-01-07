import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.optimize import curve_fit

"""

Ce fichier permet de comparer l'influence de la langue (et du traitement) sur l'apprentissage

Première comparaison : 
- un texte en francais brut et traité
- un texte en anglais brut et traité

Deuxième comparaison, 12 textes de 1.2 million de tokens :
- 3 texte en francais
- 3 texte en anglais 
- 3 texte en italien 
- 3 texte en norsk

"""

### Première comparaison

folder_test_1 = './saveStat/langue/langue_test_1/'

dicoVal = []
xmin, xmax = 1, 1800

for file in os.listdir(folder_test_1):

	l = None

	if 'fr' in file:
		if 'brut' in file : color = '#000077'; l = 'Fr [brut]'
		if 'post' in file : color = '#0000ee'; l = 'Fr [Post-T]'

	if 'en' in file:
		if 'brut' in file : color = '#770000'; l = 'En [brut]'
		if 'post' in file : color = '#ee0000'; l = 'En [Post-T]'


	if l is not None:
		with open(folder_test_1 + file, 'rb') as f:
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

	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-', label=val[-1])
	plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':') # loss value

plt.xlabel('Temps (en min)')
plt.ylabel('Loss')
plt.xscale('log')
plt.xlim([1, 25])
plt.ylim([1, 3])
plt.legend()
plt.show()



### Deuxième comparaison

folder_test_2 = './saveStat/langue/langue_test_2/'

dicoVal = []
xmin, xmax = 1, 1800

for file in os.listdir(folder_test_2):

	print(file)

	l = None

	if 'fr' in file:
		color = '#0000ff'; l = 'fr'

	if 'en' in file:
		color = '#000000'; l = 'en'

	if 'it' in file:
		color = '#00ff00'; l = 'it'

	if 'nor' in file:
		color = '#ff0000'; l = 'nor'


	if l is not None:
		with open(folder_test_2 + file, 'rb') as f:
			st = pickle.load(f)
			dicoVal.append([None, None, st.timeTV_loss, st.train_loss, st.value_loss, color, l])


def funcTheo(x, x0, a, b, c):

	return 1 / np.log(x-x0) * a + (x-x0) * b + c


plt.figure(figsize=(8, 6))

for val in dicoVal:
	xt, yt = val[2], val[3]
	xv, yv = val[2], val[4]

	stepX = np.arange(0, len(xt))

	# plt.plot(stepX*200, yt, color=val[5], marker='.', linestyle='', alpha=0.8)
	# plt.plot(stepX*200, yv, color=val[5], marker='+', linestyle='', alpha=0.8)

	popt, _ = curve_fit(funcTheo, xt, yt)
	popv, _ = curve_fit(funcTheo, xv, yv)
	X = np.linspace(xmin, xmax, 1000)

	plt.plot(X /60, funcTheo(X, *popt), color=val[5], linestyle='-')
	# plt.plot(X /60, funcTheo(X, *popv), color=val[5], linestyle=':')

plt.plot([], [], linestyle='-', color='r', label='Norsk')
plt.plot([], [], linestyle='-', color='g', label='Italian')
plt.plot([], [], linestyle='-', color='b', label='French')
plt.plot([], [], linestyle='-', color='k', label='English')

# plt.plot([], [], linestyle='-', color='r', label='Chine')
# plt.plot([], [], linestyle='-', color='g', label='Planete')
# plt.plot([], [], linestyle='-', color='b', label='Histoire')
plt.xlabel('Temps (en min)')
plt.ylabel('Loss')
# plt.xscale('log')
plt.xlim([0.5, 12])
plt.ylim([1.2,  4])
plt.legend()
plt.show()