# from time import perf_counter as ptime
from time import time as ptime
import numpy as np


class AeTime():

	"""
	Class qui permet de pouvoir visualiser la durée des fonctions et/ou méthodes de class dans un programme

	Ex : 
	# On initialise un objet Aetime :
	aetime = Aetime()

	# On peut faire :

	for i in range(1000):

		aetime.beg("total func_test")

		aetime.beg("func_test_1")
		func_test_1(i)
		aetime.end("func_test_1")

		aetime.beg("func_test_2")
		func_test_2(i)
		aetime.end("func_test_2")

		aetime.end("total func_test")

	# On regarde le résultat
	aetime.show()

	"""

	def __init__(self):

		self.dtime = {}
		self.total = ptime()
		self.totalCalcul = 0



	def beg(self, key):

		if key not in self.dtime.keys():
			# new enter dict
			self.dtime[key] = [0.0, 0.0, 0]

		self.dtime[key][1] = ptime()
		self.dtime[key][2] += 1


	def end(self, key):

		if key not in self.dtime.keys():

			print(f"warninggggg end unknowed : {key}")

		else:
			ongoing = ptime() - self.dtime[key][1]
			self.dtime[key][0] += ongoing
			self.totalCalcul += ongoing


	def show(self):

		self.total = ptime() - self.total

		print(f"Total time svg : {self.total}")
		print(f"Total total Calcul : {self.totalCalcul}")

		for key in self.dtime.keys():

			print(f"\tFor {key} : {self.dtime[key][0]} sec | {self.dtime[key][2]} time | {np.round(self.dtime[key][0]/self.total*100)} %")