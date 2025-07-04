## Projet de stage – Communications Sémantiques avec Apprentissage Profond
Ceci est une implémentation du système Deep Learning Enabled Semantic Communication Systems.

## Prérequis
Consultez le fichier requirements.txt pour la liste des bibliothèques Python nécessaires, puis installez-les avec la commande suivante :
```shell
pip install -r requirements.txt
  
## prétraitement des données
Les données utilisées sont des fichiers .txt en anglais issus des débats du Parlement européen (corpus Europarl).
Le script preprocess_text.py nettoie les textes, filtre les phrases, et transforme les données en formats .pkl et .json (encodage des phrases et vocabulaire).
Le fichier check_data.py permet d’explorer les fichiers générés dans data/.

```shell
python preprocess_text.py
