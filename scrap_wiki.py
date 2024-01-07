import wikipedia as wp
import os
from colorama import Fore, Back, Style


"""
Fichier permettant de créer des contenant de plusieurs langues en utilisant des articles wikipedia
"""

# Langues que l'on souhaite
langues = ['fr', 'en', 'it', 'no']

# Pour chaque langues, jusu'a quelles rupriques on garde le contenu de chaque articles
# On en met plusieurs car elle ne sont pas toujours dans le même ordre
cutwiki = {'fr' : ['== Notes et références ==', '== Voir aussi ==', '=== Références ===', '=== Bibliographie ===', '=== Liens externes ==='], 
		   'en' : ['== See also ==', '== Further reading ==', '== Bibliography ==', '== External links ==', '== References =='], 
		   'it' : ['== Note ==', '== Bibliografia ==', '=== Titoli generali ===', '== Voci correlate ==', '== Collegamenti esterni ==', '== Altri progetti =='], 
		   'no' : ['== Noter ==', '== Referanser ==', '=== Artikler ===', '== Eksterne lenker ==', '=== Nettsider ===', '=== Bøker ===', '== Noter og referanser ==', '== Se også ==']}

# Sujet que l'on veut aborder, avec les values correspondant au sujet dans chacune des langues de la liste langues
#     Attention : il faut que les values soit des liste qui ont la même taille que langues evidement
sujets = {'planete'  : ['planete', 'planet', 'pianeta', 'planet'],
		  'histoire' : ['histoire', 'history', 'storia', 'historie'],
		  'chine'    : ['chine', 'china', 'cina', 'kina']}

# création du dossier `wiki_texte` si il n'existe pas
save_folder = 'wiki_texte'
try :
	os.mkdir(save_folder)
except:
	pass

# Nombre de tokens voulu pour chaque sujets de chaque langues
goal_tokens = 1200000
# Nombre limite d'article dans lequelle chercher du contenu
limitResult = 400

info = {}

for sujet in sujets.keys():

	print(f"Pour le sujet {Fore.RED}{sujet}{Style.RESET_ALL} : ")
	info[sujet] = {}

	for langue, name in zip(langues, sujets[sujet]):
		print(f"Langue : {Fore.BLUE}{langue} / {sujet} : {Style.RESET_ALL}")

		wp.set_lang(langue)
		pages = wp.search(name, results=limitResult)
		# print(f"Liste pages : {pages}")

		nbtokens = 0
		nblost = 0
		nberror = 0
		total_content = ''
		current_page = 0

		while nbtokens < goal_tokens and current_page < limitResult:
			page = pages[current_page]
			try:
				print(f"{Back.RED}", end='')
				wpage = wp.page(title=page).content
				print(f"{Style.RESET_ALL}")

				nb = len(wpage)
				print(f"[{langue}|{sujet}] Page {page} : {nb} tokens before")

				sp = [spi for spi in wpage.split('\n') if len(spi) > 1]
				for cut in cutwiki[langue]:
					if cut in sp:
						print(f"{Fore.LIGHTRED_EX}CUT {cut} : {len(sp)} -> ", end='')
						sp = sp[:sp.index(cut)]
						print(f"{len(sp)}{Style.RESET_ALL}")
				sp = [spi for spi in sp if len(spi) > 40 and '==' not in spi]

				newwpage = '\n'.join(sp)
				nnb = len(newwpage)
				nblost += nb - nnb

				nbtokens += nnb
				total_content += newwpage

				print(f"    -> {nnb} tokens after [TOTAL : {nbtokens}]")
			except:
				print(f"{Back.RED}Error with {page}{Style.RESET_ALL}")
				nberror += 1
			current_page += 1

		with open(f"./{save_folder}/{sujet}_{langue}.txt", 'w', encoding='utf-8') as f:
			finaltexte = total_content[:min(goal_tokens, nbtokens)]
			truenbtokens = len(finaltexte)
			f.write(finaltexte)
			f.close()

		info[sujet][langue] = [current_page+1, nbtokens, nblost, nberror, truenbtokens]

		print(f"Final : {Fore.LIGHTBLUE_EX}{nbtokens} tokens for {langue} ... [{nblost} lost ({nblost/(nbtokens+nblost)*100:.2f}%)]{Style.RESET_ALL}")



for infokey, infoval in info.items():

	print(f"\nPour le sujet {Fore.BLUE}{infokey}{Style.RESET_ALL} : ")

	for valkey, valval in infoval.items():
		pc = valval[1] / (valval[1] + valval[2]) * 100
		print(f"    {valkey} : {Fore.GREEN}{valval[0]-valval[3]:4.0f}{Style.RESET_ALL}/{valval[0]:4.0f} pages | {valval[1]:7.0f} tokens ({pc:5.2f}%) [{Fore.GREEN}{valval[4]:7.0f} save{Style.RESET_ALL}] | {Back.RED}{valval[3]:3.0f} erreur(s){Style.RESET_ALL}")


			