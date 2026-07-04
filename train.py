import argparse
import os
import sys
import joblib
from datetime import datetime as dt
import json

import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.optim import Adam, lr_scheduler

from models import Graph2Edits
from models.model_utils import CSVLogger, get_seq_edit_accuracy
from utils.datasets import RetroEditDataset, RetroEvalDataset
from utils.mol_features import ATOM_FDIM, BOND_FDIM
from utils.rxn_graphs import Vocab

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')
ROOT_DIR = '/data/Students/Wen-Hao/Zhang-Chao/Graph2Edits-master/'

def build_model_config(args):
    model_config = {}
    if args.get('use_rxn_class', False):
        atom_fdim = ATOM_FDIM + 10
    else:
        atom_fdim = ATOM_FDIM
    model_config['n_atom_feat'] = atom_fdim
    if args.get('atom_message', False):
        model_config['n_bond_feat'] = BOND_FDIM
    else:
        model_config['n_bond_feat'] = atom_fdim + BOND_FDIM
    model_config['mpn_size'] = args['mpn_size']
    model_config['mlp_size'] = args['mlp_size']
    model_config['depth'] = args['depth']
    model_config['dropout_mlp'] = args['dropout_mlp']
    model_config['dropout_mpn'] = args['dropout_mpn']
    model_config['atom_message'] = args['atom_message']
    model_config['use_attn'] = args['use_attn']
    model_config['n_heads'] = args['n_heads']
    model_config['direction_mode'] = args.get('direction_mode', 'none')
    # Ablation knobs
    model_config['alpha_input'] = args.get('alpha_input', 'none')
    model_config['alpha_apply_mode'] = args.get('alpha_apply_mode', 'none')
    
    # New encoder parameters
    model_config['encoder_type'] = args['encoder_type']
    model_config['use_rse'] = args['use_rse']

    return model_config


def setup_device(gpu_id=None):
    """Setup CUDA device with optional GPU selection."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return 'cpu'
    
    if gpu_id is not None:
        if gpu_id >= torch.cuda.device_count():
            print(f"GPU {gpu_id} not available, using GPU 0")
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = 'cuda'
        print(f"Using default GPU: {torch.cuda.get_device_name()}")
    
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, path, is_best=False):
    save_dict = {
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'saveables': model.get_saveables() if hasattr(model, 'get_saveables') else None
    }

    if is_best:
        name = f'epoch_{epoch + 1}.pt'
    else:
        name = f'checkpoint_epoch_{epoch + 1}.pt'
    save_file = os.path.join(path, name)
    torch.save(save_dict, save_file)
    print(f'Checkpoint saved: {save_file}')


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and restore training state."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['state'])
    
    # Load optimizer state
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state restored")
    
    # Load scheduler state
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Scheduler state restored")
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {start_epoch + 1}")
    
    return start_epoch


def train_epoch(args, epoch, model, train_data, loss_fn, optimizer, device):
    torch.cuda.empty_cache()
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(train_data):
        graph_seq_tensors, seq_labels, seq_mask = batch_data
        seq_mask = seq_mask.to(device)
        seq_edit_scores = model(graph_seq_tensors)

        max_seq_len, batch_size = seq_mask.size()
        seq_loss = []

        for idx in range(max_seq_len):
            edit_labels_idx = model.to_device(seq_labels[idx])
            loss_batch = [seq_mask[idx][i] * loss_fn(seq_edit_scores[idx][i].unsqueeze(0),
                                                     torch.argmax(edit_labels_idx[i]).unsqueeze(0).long()).sum()
                          for i in range(batch_size)]

            loss = torch.stack(loss_batch, dim=0).mean()
            seq_loss.append(loss)

        total_loss = torch.stack(seq_loss).mean()
        accuracy = get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)

        train_loss += total_loss.item()
        train_acc += accuracy

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()

        if (batch_id + 1) % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss: %.4f, accuracy: %.4f' % (epoch + 1, args['epochs'], batch_id + 1, len(
                train_data), train_loss/(batch_id + 1), train_acc/(batch_id + 1)), end='', flush=True)

    train_loss = float('%.4f' % (train_loss/len(train_data)))
    train_acc = float('%.4f' % (train_acc/len(train_data)))
    print('\nepoch %d/%d, train loss: %.4f, train accuracy: %.4f' %
          (epoch + 1, args['epochs'], train_loss, train_acc))

    return train_loss, train_acc


def test(model, valid_data):
    model.eval()
    total_accuracy = 0.0
    first_step_accuracy = 0.0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(valid_data):
            prod_smi_batch, edits_batch, edits_atom_batch, rxn_classes = batch_data
            for idx, prod_smi in enumerate(prod_smi_batch):
                if rxn_classes is None:
                    edits, edits_atom = model.predict(prod_smi)
                else:
                    edits, edits_atom = model.predict(
                        prod_smi, rxn_class=rxn_classes[idx])
                if edits == edits_batch[idx] and edits_atom == edits_atom_batch[idx]:
                    total_accuracy += 1.0
                if edits[0] == edits_batch[idx][0] and edits_atom[0] == edits_atom_batch[idx][0]:
                    first_step_accuracy += 1.0
    valid_acc = float('%.4f' % (total_accuracy/len(valid_data)))
    valid_first_step_acc = float(
        '%.4f' % (first_step_accuracy/len(valid_data)))

    return valid_acc, valid_first_step_acc


def main(args):
    # Setup device
    device = setup_device(args.get('gpu_id'))
    
    # 打印训练配置参数
    print('================ Training Args ================')
    try:
        print(json.dumps(args, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception:
        print(str(args))
    print('===============================================')

    # Build experiment output directory; include ablation tags
    rxn_tag = 'with_rxn_class' if args.get('use_rxn_class', False) else 'without_rxn_class'
    ablate_tag = f"alpha_{args.get('alpha_input', 'none')}__atom_{args.get('alpha_apply_mode', 'none')}"
    out_dir = os.path.join(
        ROOT_DIR,
        'experiments',
        args['dataset'],
        rxn_tag,
        ablate_tag,
        DATE_TIME,
    )
    os.makedirs(out_dir, exist_ok=True)

    logs_filename = os.path.join(out_dir, 'logs.csv')
    csv_logger = CSVLogger(
        args=args,
        fieldnames=['epoch', 'train_acc', 'valid_acc',
                    'valid_first_step_acc', 'train_loss'],
        filename=logs_filename,
    )

    data_dir = os.path.join(ROOT_DIR, 'data', args['dataset'])
    # load bond, atom and lg vocab
    bond_vocab_file = os.path.join(data_dir, 'train', 'bond_vocab.txt')
    atom_vocab_file = os.path.join(data_dir, 'train', 'atom_lg_vocab.txt')
    bond_vocab = Vocab(joblib.load(bond_vocab_file))
    atom_vocab = Vocab(joblib.load(atom_vocab_file))

    if args.get('use_rxn_class', False):
        train_dir = os.path.join(data_dir, 'train', 'with_rxn_class')
    else:
        train_dir = os.path.join(data_dir, 'train', 'without_rxn_class')
    eval_dir = os.path.join(data_dir, 'valid')

    train_dataset = RetroEditDataset(data_dir=train_dir)
    train_data = train_dataset.loader(
        batch_size=1, num_workers=args['num_workers'], shuffle=True)

    valid_dataset = RetroEvalDataset(
        data_dir=eval_dir, data_file='valid.file.kekulized', use_rxn_class=args['use_rxn_class'])
    valid_data = valid_dataset.loader(
        batch_size=1, num_workers=args['num_workers'])

    model_config = build_model_config(args)

    model = Graph2Edits(config=model_config, atom_vocab=atom_vocab,
                        bond_vocab=bond_vocab, device=device)

    print(f'Converting model to device: {device}')
    sys.stdout.flush()
    model.to(device)
    print("Param Count: ", sum([x.nelement()
          for x in model.parameters()]) / 10**6, "M")
    print()

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'], threshold_mode='abs')

    # Handle resume from checkpoint
    start_epoch = 0
    best_acc = 0
    if args.get('resume'):
        start_epoch = load_checkpoint(args['resume'], model, optimizer, scheduler, device)
        # Load best accuracy from logs if available
        logs_file = os.path.join(out_dir, 'logs.csv')
        if os.path.exists(logs_file):
            try:
                import pandas as pd
                df = pd.read_csv(logs_file)
                if not df.empty:
                    best_acc = df['valid_acc'].max()
                    print(f"Loaded best accuracy from logs: {best_acc}")
            except:
                print("Could not load best accuracy from logs, starting from 0")

    for epoch in range(start_epoch, args['epochs']):
        train_loss, train_acc = train_epoch(
            args, epoch, model, train_data, loss_fn, optimizer, device)

        valid_acc, valid_first_step_acc = test(model, valid_data)
        scheduler.step(valid_acc)
        print('epoch %d/%d, validation accuracy: %.4f, validation_first_acc: %.4f' %
              (epoch + 1, args['epochs'], valid_acc, valid_first_step_acc))
        print('---------------------------------------------------------')
        print()

        row = {
            'epoch': str(epoch + 1),
            'train_acc': str(train_acc),
            'valid_acc': str(valid_acc),
            'valid_first_step_acc': str(valid_first_step_acc),
            'train_loss': str(train_loss),
        }
        csv_logger.writerow(row)

        # Save checkpoint every epoch (for resume capability)
        save_checkpoint(model, optimizer, scheduler, epoch, out_dir, is_best=False)
        
        # update the best accuracy for saving checkpoints
        if valid_acc >= best_acc:
            print(
                f'Best eval accuracy so far. Saving best model from epoch {epoch + 1} (acc={valid_acc})')
            print('---------------------------------------------------------')
            print()
            save_checkpoint(model, optimizer, scheduler, epoch, out_dir, is_best=True)
            best_acc = valid_acc

    csv_logger.close()
    print('Experiment finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='uspto_50k',
                        help='dataset: uspto_50k or uspto_full')
    parser.add_argument('--use_rxn_class', default=False,
                        action='store_true', help='Whether to use rxn_class')
    parser.add_argument('--atom_message', default=False, action='store_true',
                        help='Node-level or Bond-level message passing')
    parser.add_argument('--use_attn', default=False,
                        action='store_true', help='Whether to use global attention')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of heads in Multihead attention')
    parser.add_argument('--encoder_type', type=str, default='mpn',
                        choices=['mpn', 'gat', 'unfolded_gat'], 
                        help='Encoder type: mpn (original) or gat (ABANet-style)')
    parser.add_argument('--direction_mode', type=str, default='none',
                        choices=['none', 'explicit_difference', 'parity_linear'],
                        help='How to initialize h_{i→j} and h_{j→i}: none | two_linear | parity_linear')
    parser.add_argument('--use_rse', default=False, action='store_true',
                        help='Reaction State Embedding')
    # Ablation controls
    parser.add_argument('--alpha_input', type=str, default='none',
                        choices=['none', 'el', 'e0_el', 'scalar_gate_e0_el', 'vector_gate_e0_el'],
                        help='Input choice for α attention: use h^(0)/h^(l-1) and edge h^(0)/h^(l-1)')
    parser.add_argument('--alpha_apply_mode', type=str, default='none',
                        choices=['none', 'h_attn', 'm_attn', 'out_attn', 'm_out_attn'],
                        help='How to compute h_i^{(l)} from messages: concat | residual | gru')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Maximum number of epochs for training')
    parser.add_argument('--mpn_size', type=int,
                        default=256, help='MPN hidden_dim')
    parser.add_argument('--depth', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--dropout_mpn', type=float,
                        default=0.15, help='MPN dropout rate')
    parser.add_argument('--mlp_size', type=int,
                        default=512, help='MLP hidden_dim')
    parser.add_argument('--dropout_mlp', type=float,
                        default=0.2, help='MLP dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which lr will be reduced')
    parser.add_argument('--factor', type=float, default=0.8,
                        help='Factor by which the lr will be reduced')
    parser.add_argument('--thresh', type=float, default=0.01,
                        help='Threshold for measuring the new optimum')
    parser.add_argument('--max_clip', type=float, default=10.0,   
                        help='Maximum number of gradient clip')
    parser.add_argument('--print_every', type=int,
                        default=10, help='Print during train process')
    parser.add_argument('--num_workers', default=0,
                        help='Number of processes for data loading')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (0, 1, 2, etc.). If not specified, uses default GPU')

    args = parser.parse_args().__dict__
    main(args)
