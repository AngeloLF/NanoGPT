# README 

## NanoGPT personnalisé

Programmation d'un NanoGPT, encadrée par Julien Velcin, dans le cadre du premier semestre de M1 Informatique Lyon 2 pour l'UE de Initiation Recherche.

## Fonctionnement

Le programme central est dans le fichier `nanoGPTv3v2.py`, et un exemple d'utilisation est donné dans le fichier `mainNanoGPT.py`.

On a plusieurs dossiers qui stockent différentes sortes de fichiers : 
* Le dossier `source` permet de stocker les textes (`.txt`) qui servent à l'apprentissage des modèles.
* Le dossier `saveStat` permet de stocker les fichiers `.stat` qui servent à l'analyse
* Le dossier `saveGPT` permet de stocker une sauvegarde des modèles, pour les utiliser plus tard
  
Ensuite, on a différents programmes que l'on peut utiliser après la création des modèles :
* `generationGPT.py` : permet de générer du texte à partir de modèle `.gpt` sauvegardé
* Tout les programmes en `stats_***.py` qui permettent de faire une analyse des différents hyperparamètres 

## Package nécessaire

Pour que notre programme s'exécute correctement, nous avons besoin des packages suivants : 
* os, sys, time, pickle
* numpy, matplotlib, scipy
* torch
* colorama
* wikipedia

Date : 01/2024
