import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data
from utils import greedy_decode, SeqtoText, SNR_to_noise, saturated_amplifier_with_ibo_real
from models.transceiver import DeepSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger vocabulaire
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
token_to_idx = vocab['token_to_idx']
padding_idx = token_to_idx['<PAD>']
start_symbol = token_to_idx['<START>']
end_idx = token_to_idx['<END>']

seq_to_text = SeqtoText(token_to_idx, end_idx)

# Charger dataset test complet
test_dataset = EurDataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_data)

# Paramètres modèle
num_layers = 4
src_vocab_size = len(token_to_idx)
trg_vocab_size = len(token_to_idx)
d_model = 128
num_heads = 8
dff = 512
dropout = 0.1

model = DeepSC(num_layers, src_vocab_size, trg_vocab_size, src_vocab_size, trg_vocab_size,
               d_model, num_heads, dff, dropout).to(device)

# Charger checkpoint
checkpoint_path = "checkpoints/deepsc-AWGN/checkpoint_80.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

snr_db = 12
n_var = SNR_to_noise(snr_db)
ibo_db = 3  # valeur d’IBO

# Visualiser les 3 premiers signaux
nb_figures = 3
fig_count = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        src = batch.to(device)

        # Récupération signal avant et après ampli
        outputs, tx_before_ampli, tx_after_ampli = greedy_decode(
            model, src, n_var=n_var, max_len=src.size(1),
            padding_idx=padding_idx, start_symbol=start_symbol,
            channel='AWGN',
            ibo_db=ibo_db,
            return_signals=True
        )

        # Recalcul de Asat pour affichage
        P = torch.mean(tx_before_ampli**2)
        IBO = 10 ** (ibo_db / 10)
        Asat = torch.sqrt(IBO * P).item()

        print(f"Batch {i+1}: Asat = {Asat:.4f}")

        if fig_count < nb_figures:
            signal_before = tx_before_ampli[0].cpu().numpy().flatten()
            signal_after = tx_after_ampli[0].cpu().numpy().flatten()

            plt.figure(figsize=(12, 5))
            plt.plot(signal_before, label="Avant ampli", linewidth=2)
            plt.plot(signal_after, label=f"Après ampli (Asat={Asat:.2f})", linewidth=2)
            plt.title(f"Batch {i+1} : Amplificateur saturé (IBO = {ibo_db} dB)")
            plt.xlabel("Index de la dimension")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            fig_count += 1
