import pickle
from nanoGPTv3v2 import GPTLanguageModel
from time import perf_counter as ptime


"""
Ce fichier main permet de lancer l'apprentissage avec les caractérisques voulu
Il est à modifier selon les modèles que l'on veut

Ce fichier est juste un exemple d'utilisation de la class GPTLanguageModel



"""


textes = ['source/zola/germinal_traite.txt']
total = len(textes)
label_test = 'test'

current = 0
t0 = ptime()

for texte in textes:

	batch = 32
	block = 128
	embd = 128
	max_iters = 2000
	eval_inter = 200

	current += 1
	print(f"Now : BATCH {batch} | BLOCK {block} | EMBD {embd} | MAX_ITERS {max_iters} | {texte} [{current}/{total}]")

	hyperParam = {
		'batch_size'    : batch,
		'block_size'    : block,
		'learning_rate' : 1e-3, # 5e-4
		'max_iters'     : max_iters + 1,
		'eval_interval' : eval_inter,

		'eval_iters'    : 100,
		'n_embd'        : embd,
		'n_head'        : 4,
		'n_layer'       : 4,
		'modelDropout'  : 0.2,
		'seed'          : 1337,

		'limitLearning' : None,

		# Nom du fichier de sauvegarde de stats
		'termSave'      : f"{texte.split('/')[-1][:-4]}-[{batch};{block};{embd};{max_iters}]-[{label_test}_{current}]",

		'inputData'     : texte,
		# Nom du fichier de sauvegarde du gpt
		'outputData'    : f"./saveGPT/{texte.split('/')[-1][:-4]}-[{batch};{block};{embd};{max_iters}]-[{label_test}_{current}].gpt"}

	model = GPTLanguageModel(**hyperParam)
	model.letsLearning()
	model.saveGPT()


tf = round((ptime() - t0) / 60)
th = round(tf/60, 2)
print(f"Temps total : {tf} min [{th} h]")