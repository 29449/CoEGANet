import os
import joblib
import torch
import numpy as np
import argparse
from tqdm import tqdm
from rdkit import Chem
from models import Graph2Edits, BeamSearch

ROOT_DIR = '/data/Students/Wen-Hao/Zhang-Chao/Graph2Edits-master/'

def load_model(exp_dir, pt_file, device):
    checkpoint = torch.load(os.path.join(exp_dir, pt_file), map_location=device)
    config = checkpoint['saveables']
    model = Graph2Edits(**config, device=device)
    model.load_state_dict(checkpoint['state'])
    model.to(device)
    model.eval()
    return model, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50k')
    parser.add_argument("--use_rxn_class", default=False, action='store_true')
    parser.add_argument('--alpha_input', type=str, default='none')
    parser.add_argument('--alpha_apply_mode', type=str, default='none')
    parser.add_argument('--experiments', type=str, default='27-06-2022--10-27-22')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=9)
    parser.add_argument('--pt_file', type=str, default='epoch_123.pt')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    # ===== load data =====
    data_dir = os.path.join(ROOT_DIR, 'data', f'{args.dataset}', 'test')
    test_data = joblib.load(os.path.join(data_dir, 'test.file.kekulized'))

    rxn_tag = 'with_rxn_class' if args.use_rxn_class else 'without_rxn_class'
    ablate_tag = f"alpha_{args.alpha_input}__atom_{args.alpha_apply_mode}"
    exp_dir = os.path.join(
        ROOT_DIR, 'experiments', f'{args.dataset}', rxn_tag, ablate_tag, args.experiments
    )

    model, config = load_model(exp_dir, args.pt_file, device)
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)

    # ===== statistics containers =====
    T_true_list = []
    T_pred_list = []
    early = late = correct = 0

    save_rse = True
    rse_records = []

    def log_stats(tag):
        N = len(T_true_list)
        if N == 0:
            return
        early_rate = early / N
        late_rate = late / N
        acc_rate = correct / N
        mae = np.mean(np.abs(np.array(T_true_list) - np.array(T_pred_list)))

        print("===== Termination Analysis =====")
        print(f"[Up to {tag} samples]")
        print(f"Total samples: {N}")
        print(f"Early termination rate:   {early_rate:.4f}")
        print(f"Late termination rate:    {late_rate:.4f}")
        print(f"Correct termination rate: {acc_rate:.4f}")
        print(f"Step MAE:                 {mae:.4f}")

        out_file = os.path.join(exp_dir, f'rse_termination_stats_{tag}.txt')
        with open(out_file, 'w') as f:
            f.write(f"samples\t{N}\n")
            f.write(f"early_rate\t{early_rate}\n")
            f.write(f"late_rate\t{late_rate}\n")
            f.write(f"correct_rate\t{acc_rate}\n")
            f.write(f"mae\t{mae}\n")

        if save_rse:
            np.save(os.path.join(exp_dir, f'rse_vectors_{tag}.npy'), rse_records)

    # ===== main loop =====
    for idx in tqdm(range(len(test_data))):
        rxn_data = test_data[idx]
        rxn_smi = rxn_data.rxn_smi
        rxn_class = rxn_data.rxn_class

        T_true = len(rxn_data.edits)
        T_true_list.append(T_true)
  
        r, p = rxn_smi.split('>>')

        with torch.no_grad():
            paths = beam_model.run_search(
                prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class
            )

        best_path = paths[0]
        pred_actions = best_path['rxn_actions']
        T_pred = len(pred_actions)
        T_pred_list.append(T_pred)

        if T_pred < T_true:
            early += 1
        elif T_pred > T_true:
            late += 1
        else:
            correct += 1

        # ===== record RSE =====
        if save_rse and 'rse_seq' in best_path:
            rse_seq = best_path['rse_seq']
            for t, vec in enumerate(rse_seq):
                rse_records.append({
                    'idx': idx,
                    't': t,
                    'T_true': T_true,
                    'T_pred': T_pred,
                    'rse': vec.cpu().numpy(),
                    'rxn_class': rxn_class
                })
  
        # ===== every 50 samples, log once =====
        # if (idx + 1) % 50 == 0:
        #     log_stats(tag=str(idx + 1))

    # ===== final stats =====
    log_stats(tag="final")

if __name__ == '__main__':
    main()
