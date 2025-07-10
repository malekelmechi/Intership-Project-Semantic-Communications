import os
import json
import torch
import argparse
import numpy as np
import csv
from torch.utils.data import DataLoader
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Appareil utilisé : {device}")

def performance(args, SNR_list, model, token_to_idx, pad_idx, start_idx, end_idx, ibo_db=15):
    bleu1 = BleuScore(1, 0, 0, 0)
    bleu2 = BleuScore(0.5, 0.5, 0, 0)
    bleu3 = BleuScore(0.33, 0.33, 0.33, 0)
    bleu4 = BleuScore(0.25, 0.25, 0.25, 0.25)

    test_set = EurDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_data, num_workers=0)

    StoT = SeqtoText(token_to_idx, end_idx)

    bleu1_results, bleu2_results, bleu3_results, bleu4_results = [], [], [], []

    model.eval()
    with torch.no_grad():
        for snr in tqdm(SNR_list, desc="SNR Loop"):
            noise_std = SNR_to_noise(snr)

            pred_sentences = []
            target_sentences = []

            for batch in test_loader:
                batch = batch.to(device)
                target = batch

                output = greedy_decode(model, batch, noise_std, args.MAX_LENGTH, pad_idx, start_idx,
                                       args.channel, ibo_db=ibo_db, return_signals=False)

                pred_strings = list(map(StoT.sequence_to_text, output.cpu().numpy().tolist()))
                target_strings = list(map(StoT.sequence_to_text, target.cpu().numpy().tolist()))

                pred_sentences.extend(pred_strings)
                target_sentences.extend(target_strings)

            bleu1_score = np.mean(bleu1.compute_blue_score(pred_sentences, target_sentences))
            bleu2_score = np.mean(bleu2.compute_blue_score(pred_sentences, target_sentences))
            bleu3_score = np.mean(bleu3.compute_blue_score(pred_sentences, target_sentences))
            bleu4_score = np.mean(bleu4.compute_blue_score(pred_sentences, target_sentences))

            bleu1_results.append(bleu1_score)
            bleu2_results.append(bleu2_score)
            bleu3_results.append(bleu3_score)
            bleu4_results.append(bleu4_score)

            print(f"\n[IBO {ibo_db} dB | SNR {snr} dB] BLEU-1: {bleu1_score:.4f}, BLEU-2: {bleu2_score:.4f}, BLEU-3: {bleu3_score:.4f}, BLEU-4: {bleu4_score:.4f}")

    return bleu1_results, bleu2_results, bleu3_results, bleu4_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
    parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str)
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    args = parser.parse_args()

    SNR_values = [0,3,6,9,12,15,18]
    IBO_values = [1,2,3,4,5,6]  # 💡 valeurs à tester

    vocab = json.load(open(args.vocab_file, 'r'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    num_vocab = len(token_to_idx)

    model = DeepSC(4, num_vocab, num_vocab, num_vocab, num_vocab, 128, 8, 512, 0.1).to(device)

    best_model_path = os.path.join(args.checkpoint_path, "best_model.pth")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"❌ Le fichier {best_model_path} n'existe pas.")
         
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"📦 Modèle chargé depuis {best_model_path}")

    all_results = {}

    for ibo_db in IBO_values:
        print(f"\n=== Test avec IBO = {ibo_db} dB ===")
        bleu1, bleu2, bleu3, bleu4 = performance(
            args, SNR_values, model, token_to_idx, pad_idx, start_idx, end_idx, ibo_db=ibo_db
        )
        all_results[ibo_db] = (bleu1, bleu2, bleu3, bleu4)

    # Sauvegarde CSV
    csv_file = f"fichier_bleu_modele_sans_ampli.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR', 'IBO_dB', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'])
        for ibo_db, (bleu1_list, bleu2_list, bleu3_list, bleu4_list) in all_results.items():
            for snr, b1, b2, b3, b4 in zip(SNR_values, bleu1_list, bleu2_list, bleu3_list, bleu4_list):
                writer.writerow([snr, ibo_db, b1, b2, b3, b4])

    print(f"✅ Résultats enregistrés dans {csv_file}")
