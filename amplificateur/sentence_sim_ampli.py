import os
os.environ["USE_TF"] = "0"

import json
import torch
import argparse
import numpy as np
import csv
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sentence_similarity_ampli(args, SNR_list, model, token_to_idx, pad_idx, start_idx, end_idx, ibo_db):
    StoT = SeqtoText(token_to_idx, end_idx)
    test_set = EurDataset('test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_data, num_workers=0)

    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    sim_scores_by_snr = []

    model.eval()
    with torch.no_grad():
        for snr in tqdm(SNR_list, desc=f"🔁 SNR Loop (IBO={ibo_db} dB)"):
            noise_std = SNR_to_noise(snr)
            pred_sentences, target_sentences = [], []

            for batch in test_loader:
                batch = batch.to(device)
                output = greedy_decode(model, batch, noise_std, args.MAX_LENGTH, pad_idx, start_idx,
                                       args.channel, ibo_db=ibo_db, return_signals=False)

                pred_strings = list(map(StoT.sequence_to_text, output.cpu().numpy().tolist()))
                target_strings = list(map(StoT.sequence_to_text, batch.cpu().numpy().tolist()))

                pred_sentences.extend(pred_strings)
                target_sentences.extend(target_strings)

            embeddings_pred = bert_model.encode(pred_sentences, convert_to_tensor=True)
            embeddings_target = bert_model.encode(target_sentences, convert_to_tensor=True)

            cos_sim = util.cos_sim(embeddings_pred, embeddings_target)
            sim_scores = cos_sim.diag()
            mean_score = sim_scores.mean().item()
            sim_scores_by_snr.append(mean_score)

            print(f"[SNR {snr} dB] ➤ Similarité moyenne : {mean_score:.4f}")

    return sim_scores_by_snr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
    parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
    parser.add_argument('--channel', default='Rayleigh', type=str)
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    args = parser.parse_args()

    SNR_values = [0, 3, 6, 9, 12, 15, 18]
    IBO_values = [1,2,3,4,5,6]  # à adapter

    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    num_vocab = len(token_to_idx)

    model = DeepSC(args.num_layers, num_vocab, num_vocab, num_vocab, num_vocab,
                   args.d_model, args.num_heads, args.dff, 0.1).to(device)

    best_model_path = os.path.join(args.checkpoint_path, "best_model.pth")
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"❌ Le fichier {best_model_path} n'existe pas.")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Modèle chargé depuis {best_model_path}")

    # Test avec différentes valeurs de IBO
    all_similarities = {}

    for ibo in IBO_values:
        print(f"\n=== 🚀 Test avec IBO = {ibo} dB ===")
        sim_scores = sentence_similarity_ampli(args, SNR_values, model, token_to_idx,
                                               pad_idx, start_idx, end_idx, ibo_db=ibo)
        all_similarities[ibo] = sim_scores

    # Sauvegarder dans un CSV
    csv_file = f"impact_sen_ray.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR', 'IBO_dB', 'SentenceSimilarity'])
        for ibo, scores in all_similarities.items():
            for snr, sim in zip(SNR_values, scores):
                writer.writerow([snr, ibo, sim])

    print(f"\n📁 Résultats sauvegardés dans : {csv_file}")
