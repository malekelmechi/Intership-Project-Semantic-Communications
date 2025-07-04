import pickle
import json

def decode(seq_idx, idx_to_token, delim=' ', stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        token = idx_to_token.get(idx, '<UNK>')
        tokens.append(token)
        if stop_at_end and token == '<END>':
            break
    return delim.join(tokens)

# Charger les données picklées
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Nombre de phrases dans train_data.pkl : {len(train_data)}")
print(f"Nombre de phrases dans test_data.pkl : {len(test_data)}")

# Charger vocab.json
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

token_to_idx = vocab['token_to_idx']
print(f"Taille du vocabulaire : {len(token_to_idx)}")
print(f"Tokens spéciaux dans le vocabulaire : { {k:v for k,v in token_to_idx.items() if k.startswith('<')} }")

# Inverser token_to_idx pour idx_to_token
idx_to_token = {v: k for k, v in token_to_idx.items()}

# Exemple : décoder la première phrase du train_data
exemple_indices = train_data[3]

print("Exemple de phrase encodée (indices):", exemple_indices)
print("Exemple décodé :", decode(exemple_indices, idx_to_token))
