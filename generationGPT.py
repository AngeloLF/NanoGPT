import pickle
from nanoGPTv3v2 import GPTLanguageModel
import torch
import matplotlib.pyplot as plt
import time

"""
Fichier pour généré du texte a partir de la sauvgarde d'un modèle gpt (choisir un fichier .gpt)
"""

new_tokens = 500 # Nombre de tokens à généré
context    = 'Pour' # Morceau de texte pris pour le context initiale 
outputData = 'saveGPT/germinal_petit_test.gpt' # Fichier de sauvegarde d'un gpt


with open(outputData, 'rb') as f:
    model = pickle.load(f)

model.generateFromContext(context=context, new_token=new_tokens, showStep=True, seed=1337)