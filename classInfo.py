import numpy as np
import matplotlib.pyplot as plt
import pickle

class StatGPT():

	"""
	Class permettant d'enregistrer les performances de l'apprentissage d'un modèle NanoGPT dans un fichier ".stat"
	"""


	def __init__(self, gptParam, saveFreq, termSave):

		self.gptParam = gptParam

		self.sf = saveFreq

		self.loss = []
		self.time = []

		self.stepTV_loss = []
		self.timeTV_loss = []
		self.train_loss = []
		self.value_loss = []

		self.termSave = termSave

		self.aetime = None
		self.nbParam = None

	def saveStat(self, folder='saveStat'):

		batch = self.gptParam['batch_size']
		block = self.gptParam['block_size']
		learn = self.gptParam['learning_rate']
		n_emb = self.gptParam['n_embd']

		with open(f"./{folder}/{self.termSave}.stat", 'wb') as f:
			pickle.dump(self, f)


	def plotStat(self):

		plt.plot(self.time, self.loss, color='r', marker='.')
		plt.plot(self.timeTV_loss, self.train_loss, color='#00bb00', marker='.', linestyle=':')
		plt.plot(self.timeTV_loss, self.value_loss, color='#00ff00', marker='.', linestyle=':')
		plt.show()

