# Intership-Project-Semantic-Communications

This is the implementation of  Deep learning enabled semantic communication systems.

## Requirements
+ See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.
  
## prétraitement des données
Les données utilisées sont des fichiers .txt en anglais issus des débats du Parlement européen (corpus Europarl).
Le script preprocess_text.py nettoie les textes, filtre les phrases, et transforme les données en formats .pkl et .json (encodage des phrases et vocabulaire).
Le fichier check_data.py permet d’explorer les fichiers générés dans data/.

```shell
python preprocess_text.py
