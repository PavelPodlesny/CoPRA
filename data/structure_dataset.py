from data.register import DataRegister
from torch.utils.data import Dataset
import pandas as pd
import esm
import torch
from tqdm import tqdm
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
from tqdm import tqdm
import os
import math
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from data.prepare import ComplexData

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()


@R.register('structure_dataset')
class StructureDataset(Dataset):
    '''
    Loads a whole partition (train/val/test) of already-`precache`d
    `ComplexData` files into memory. No PDB parsing, no distance-map
    computation, no diskcache — all of that happens offline, once, via
    `python run.py precache` (see `data/prepare.py`).
    '''
    def __init__(self,
                 dataframe,
                 prepared_dir,
                 col_prot_name='PDB',
                 transform=None,
                 **kwargs
                 ):
        self.prepared_dir = prepared_dir
        self.col_prot_name = col_prot_name
        self.transform = get_transform(transform)

        self.data = []
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            structure_id = row[col_prot_name]
            path = os.path.join(prepared_dir, f'{structure_id}.pkl')
            self.data.append(ComplexData.load(path).decompress())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].to_dict()
        if self.transform is not None:
            data = self.transform(data)
        return data

EXCLUDE_KEYS = ['labels', 'complex']
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mask_atoms': 0,
    'chain_nb': -1,
}
PAIRWISE_2D_KEYS = ['pairwise_dist', 'pairwise_dihedral', 'interface_energy']

class CustomStructCollate(object):
    def __init__(self, strategy='separate', length_ref_key='restype', pad_values=DEFAULT_PAD_VALUES, exclude_keys=EXCLUDE_KEYS, eight=True):
        super().__init__()
        self.strategy = strategy
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.exclude_keys = exclude_keys
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _pad_2d(x, n, value=0):
        # Pads dims 0 AND 1 to n (x is square in its first two dims: the
        # precomputed pairwise geometry/InNA-energy tensors from work-stream B).
        if not isinstance(x, torch.Tensor):
            return x
        l = x.size(0)
        assert x.size(1) == l and l <= n
        if l == n:
            return x
        pad_shape = (n, n) + tuple(x.shape[2:])
        padded = torch.full(pad_shape, fill_value=value, dtype=x.dtype, device=x.device)
        padded[:l, :l, ...] = x
        return padded

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_2d = []
        keys_not_pad = []
        keys_ignore = ['prot_seqs', 'rna_seqs', 'mut_seqs', 'max_prot_length', 'max_na_length', 'atom_min_dist']
        for key in keys_inter:
            if key in keys_ignore:
                continue
            elif key in PAIRWISE_2D_KEYS:
                keys_2d.append(key)
            elif key not in self.exclude_keys:
                keys.append(key)
            else:
                keys_not_pad.append(key)

        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []

        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            for k in keys_2d:
                data_padded[k] = self._pad_2d(data[k], max_length, value=0)
            for k in keys_not_pad:
                data_padded[k] = data[k]
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_berts(self, strategy, batch):
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]

        max_item_prot_length = [item['max_prot_length'] for item in batch]
        max_item_na_length = [item['max_na_length'] for item in batch]
        max_prot_length = max(max_item_prot_length)
        max_na_length = max(max_item_na_length)
        total_prot_chains = sum(prot_chains)
        total_na_chains = sum(na_chains)
        if self.eight:
            max_prot_length = math.ceil((max_prot_length + 2) / 8) * 8
            max_na_length =  math.ceil((max_na_length + 2) / 8) * 8
        else:
            max_prot_length = max_prot_length + 2
            max_na_length = max_na_length + 2
        prot_batch = torch.empty([total_prot_chains, max_prot_length])
        prot_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            na_seqs = item['rna_seqs']
            for i, prot_seq in enumerate(prot_seqs):
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                curr_prot_idx += 1
            for na_seq in na_seqs:
                # na_batch[curr_na_idx, 0] = na_alphabet.cls_idx
                # NA encoder adds CLS and EOS by default
                na_seq_encode = na_alphabet.encode(na_seq)
                seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                na_batch[curr_na_idx, :len(seq)] = seq
                # na_batch[curr_na_idx, len(na_seq_encode)+1] = na_alphabet.eos_idx
                curr_na_idx += 1
        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        return prot_batch.long(), prot_chains, prot_mask, na_batch.long(), na_chains, na_mask

    def __call__(self, data_list):
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        prot_batch, prot_chains, prot_mask, na_batch, na_chains, na_mask = self.pad_for_berts(self.strategy, data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        batch['strategy'] = self.strategy
        batch['labels'] = batch['labels'].float()
        return batch
