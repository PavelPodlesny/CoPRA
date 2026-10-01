"""Offline data preparation for CoPRA (dG only).

Parses each complex's PDB once, computes everything that's model-independent
(structure arrays, raw pairwise backbone-atom geometry, and a frozen-InNA
per-residue-pair interface-energy map), and writes one `ComplexData` file per
structure under `prepared_dir`. Run via `python run.py precache` *before*
training — `StructureDataset` (`data/structure_dataset.py`) then just loads
these files, no PDB parsing at train time.

Everything offline lives in this one file for easy end-to-end review.
"""
import dataclasses
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from data.complex import ComplexInput
from data.transforms.select_atom import SelectAtom
from utils.geometry import angstrom_to_nm, pairwise_dihedrals


# ---------------------------------------------------------------------------
# ComplexData: the per-structure, disk-persisted representation
# ---------------------------------------------------------------------------

@dataclass
class ComplexData:
    seq: str
    prot_seqs: list
    rna_seqs: list
    res_nb: torch.Tensor
    chain_nb: torch.Tensor
    identifier: torch.Tensor
    restype: torch.Tensor
    seq_mask: torch.Tensor
    pos_heavyatom: torch.Tensor
    mask_heavyatom: torch.Tensor
    atom64_positions: torch.Tensor
    atom64_mask: torch.Tensor
    atom_min_dist: torch.Tensor      # (L, L) — used by selected_region_with_distmap
    pairwise_dist: torch.Tensor      # (L, L, 16) — raw backbone-atom distances
    pairwise_dihedral: torch.Tensor  # (L, L, 2)  — raw phi/psi dihedrals
    interface_energy: torch.Tensor   # (L, L) — raw InNA per-residue-pair energy
    energy_mask: torch.Tensor        # (L, L) bool — True where InNA actually produced interface_energy
    max_prot_length: int
    max_na_length: int
    structure_id: str
    label: float

    def compress(self) -> "ComplexData":
        return dataclasses.replace(
            self,
            restype=self.restype.to(torch.int8),
            chain_nb=self.chain_nb.to(torch.int8),
            identifier=self.identifier.to(torch.int8),
            res_nb=self.res_nb.to(torch.int16),
        )

    def decompress(self) -> "ComplexData":
        return dataclasses.replace(
            self,
            restype=self.restype.to(torch.int64),
            chain_nb=self.chain_nb.to(torch.int64),
            identifier=self.identifier.to(torch.int64),
            res_nb=self.res_nb.to(torch.int64),
        )

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.compress(), f)

    @classmethod
    def load(cls, path) -> "ComplexData":
        with open(path, 'rb') as f:
            return pickle.load(f)

    def to_dict(self) -> dict:
        # Fresh dict each call, shallow tensor references — transforms may
        # add/overwrite keys on it without mutating this ComplexData.
        return {
            'seq': self.seq,
            'prot_seqs': self.prot_seqs,
            'rna_seqs': self.rna_seqs,
            'res_nb': self.res_nb,
            'chain_nb': self.chain_nb,
            'identifier': self.identifier,
            'restype': self.restype,
            'seq_mask': self.seq_mask,
            'pos_heavyatom': self.pos_heavyatom,
            'mask_heavyatom': self.mask_heavyatom,
            'atom64_positions': self.atom64_positions,
            'atom64_mask': self.atom64_mask,
            'atom_min_dist': self.atom_min_dist,
            'pairwise_dist': self.pairwise_dist,
            'pairwise_dihedral': self.pairwise_dihedral,
            'interface_energy': self.interface_energy,
            'energy_mask': self.energy_mask,
            'max_prot_length': self.max_prot_length,
            'max_na_length': self.max_na_length,
            'labels': self.label,
        }


# ---------------------------------------------------------------------------
# Raw geometry precomputation (moved out of ResiduePairEncoder.forward)
# ---------------------------------------------------------------------------

def _compute_atom_min_dist(coords, mask):
    L = coords.shape[0] # (L, 41, 3)
    distance_map = torch.linalg.norm(
        coords[:, None, :, None, :] - coords[None, :, None, :, :], dim=-1, ord=2
    ).reshape(L, L, -1) # (L, L, 41, 41, 3) -> (L, L, 41, 41) -> (L, L, 41*41)
    mask = (mask[:, None, :, None] * mask[None, :, None, :]).reshape(L, L, -1) # 0 ~ missing pair
    distance_map[~mask] = torch.inf
    return torch.min(distance_map, dim=-1)[0] # (values, indices)[0] -> (L, L)


def _compute_pairwise_geometry(pos_heavyatom, mask_heavyatom, identifier, seq, atom_resolution='backbone'):
    """Reuses the SelectAtom transform to derive pos_atoms/mask_atoms exactly
    as training would (same atom_resolution), then precomputes the raw
    pairwise backbone-atom distances and phi/psi dihedrals that
    ResiduePairEncoder used to compute on every forward pass."""
    backbone = SelectAtom(resolution=atom_resolution)({
        'pos_heavyatom': pos_heavyatom,
        'mask_heavyatom': mask_heavyatom,
        'identifier': identifier,
        'seq': seq,
    })
    pos_atoms = backbone['pos_atoms']  # (L, A, 3)
    L = pos_atoms.shape[0]

    pairwise_dist = angstrom_to_nm(torch.linalg.norm(
        pos_atoms[:, None, :, None] - pos_atoms[None, :, None, :],
        dim=-1, ord=2,
    )).reshape(L, L, -1) # (L, L, A*A)
    pairwise_dihedral = pairwise_dihedrals(pos_atoms.unsqueeze(0)).squeeze(0)  # (L, L, 2)
    return pairwise_dist, pairwise_dihedral


# ---------------------------------------------------------------------------
# InNA-derived interface energy (frozen, offline only)
# ---------------------------------------------------------------------------

def load_inna_model(weights_path, repo_path=None, device='cpu'):
    """One-time load of the frozen InNA model. `repo_path` is inserted into
    sys.path so `model.Inna`/`model.dataset` (the sibling InNA repo's own
    package layout) can be imported."""
    if repo_path is not None and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from model.Inna import InNA
    model = InNA.load(weights_path)
    model.to(device)
    model.eval()
    return model


def compute_interface_energy_map(cplx, inna_model, device='cpu', batch_size=256):
    """Raw (unembedded) InNA interaction energy for every protein-residue x
    RNA-residue pair in the complex — the same universe of pairs
    ResiduePairEncoder builds features for, no distance cutoff. Same-molecule
    entries and the diagonal are left at 0.0 (InNA has no notion of them).

    Builds InNA's per-pair inputs directly from `cplx` (data/inna_helpers.py),
    not by re-parsing the PDB via naskit — this also sidesteps naskit's own
    chain-traversal-order and residue-numbering-gap bugs (see the historical
    note in data/inna_helpers.py) entirely, since `cplx` already carries the
    correctly chain-ordered, gap-safe residue layout `ComplexInput` built.

    Returns (energy_map, energy_mask, info): energy_map and energy_mask are
    (L, L), L = cplx.restype.shape[0]. energy_mask is True only where InNA
    produced a value, so an uncomputed pair (left at 0.0 in energy_map) can be
    told apart from a real energy near 0.
    info = {'status', 'pairs_total', 'pairs_skipped'}: status is always 'ok'
    here (cplx already parsed successfully by the time this runs); pairs_total
    is n_protein_residues * n_rna_residues in the whole complex, pairs_skipped
    counts how many of those were left at zero because a residue was entirely
    unresolved (every one of its atoms missing) or unknown to InNA -- a
    residue with only *some* atoms missing (e.g. a truncated side chain) is
    still scored, by design (data/inna_helpers.py's skip policy)."""
    from model.dataset import collate_fn
    from data.inna_helpers import complex_to_inna_pairs

    L = cplx.restype.shape[0]
    energy_map = torch.zeros(L, L)
    energy_mask = torch.zeros(L, L, dtype=torch.bool)
    n_prot = int((cplx.identifier == 0).sum())
    n_rna = int((cplx.identifier == 1).sum())
    info = {'status': 'ok', 'pairs_total': n_prot * n_rna, 'pairs_skipped': 0}

    pairs, indices = complex_to_inna_pairs(cplx)
    info['pairs_skipped'] = info['pairs_total'] - len(pairs)

    for start in range(0, len(pairs), batch_size):
        chunk_items = pairs[start:start + batch_size]
        chunk_idx = indices[start:start + batch_size]
        batch = collate_fn(chunk_items).to(device)
        with torch.no_grad():
            energies = inna_model.predict(batch).energy.cpu()
        for (i, j), e in zip(chunk_idx, energies):  # (i, j) are already global row indices
            energy_map[i, j] = e.item()
            energy_map[j, i] = e.item()
            energy_mask[i, j] = True
            energy_mask[j, i] = True

    return energy_map, energy_mask, info


# ---------------------------------------------------------------------------
# Per-structure preparation and the precache driver
# ---------------------------------------------------------------------------

def prepare_complex(data_root,
                    cplx_id, prot_chains, na_chains, energy,
                    inna_model=None, atom_resolution='backbone', device='cpu', **kwargs
                     ) -> Tuple[Optional[ComplexData], Optional[dict]]:
    """Returns (ComplexData, energy_info), or (None, None) if the structure can't be parsed.
    energy_info is compute_interface_energy_map's info dict, with status
    'inna_disabled' when no InNA model was given (all-zero energy map)."""

    pdb_path = os.path.join(data_root, cplx_id + '.pdb')

    cplx = ComplexInput.from_path(pdb_path, valid_prot_chains=prot_chains, valid_rna_chains=na_chains)
    if cplx is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {pdb_path}')
        return None, None

    res_nb = torch.LongTensor(cplx.res_nb)
    chain_nb = torch.LongTensor(cplx.chainid)
    identifier = torch.LongTensor(cplx.identifier)
    restype = torch.LongTensor(cplx.restype)
    seq_mask = torch.BoolTensor(cplx.mask)
    pos_heavyatom = torch.FloatTensor(cplx.atom41_positions)
    mask_heavyatom = torch.BoolTensor(cplx.atom41_mask)
    atom64_positions = torch.FloatTensor(cplx.atom_positions)
    atom64_mask = torch.BoolTensor(cplx.atom_mask)

    atom_min_dist = _compute_atom_min_dist(pos_heavyatom, mask_heavyatom)
    pairwise_dist, pairwise_dihedral = _compute_pairwise_geometry(
        pos_heavyatom, mask_heavyatom, identifier, cplx.seq, atom_resolution=atom_resolution,
    )

    if inna_model is not None:
        interface_energy, energy_mask, energy_info = compute_interface_energy_map(
            cplx, inna_model, device=device,
        )
    else:
        L = len(cplx.seq)
        interface_energy = torch.zeros(L, L)
        energy_mask = torch.zeros(L, L, dtype=torch.bool)
        energy_info ={'status': 'inna_disabled', 'pairs_total': 0, 'pairs_skipped': 0}

    max_prot_length = max((len(s) for s in cplx.prot_seqs), default=0)
    max_na_length = max((len(s) for s in cplx.na_seqs), default=0)

    return ComplexData(
        seq=cplx.seq,
        prot_seqs=cplx.prot_seqs,
        rna_seqs=cplx.na_seqs,
        res_nb=res_nb,
        chain_nb=chain_nb,
        identifier=identifier,
        restype=restype,
        seq_mask=seq_mask,
        pos_heavyatom=pos_heavyatom,
        mask_heavyatom=mask_heavyatom,
        atom64_positions=atom64_positions,
        atom64_mask=atom64_mask,
        atom_min_dist=atom_min_dist,
        pairwise_dist=pairwise_dist,
        pairwise_dihedral=pairwise_dihedral,
        interface_energy=interface_energy,
        energy_mask=energy_mask,
        max_prot_length=max_prot_length,
        max_na_length=max_na_length,
        structure_id=cplx_id,
        label=energy,
    ), energy_info


def _print_energy_report(report, n_cached):
    """Summary of where zeros were written as a fallback instead of a computed
    InNA energy. `report` only covers structures processed in this run."""
    print(f'\n[energy report] processed {len(report)} structures this run '
          f'({n_cached} already cached, not audited)')
    by_status = {}
    for r in report:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    for status, n in sorted(by_status.items()):
        print(f'  {status}: {n}')
    partial = [r for r in report if r['status'] == 'ok' and r['pairs_skipped'] > 0]
    print(f'  ok but with skipped pairs: {len(partial)}')
    for r in report:
        if r['status'] != 'ok':
            print(f"  [{r['status']}] {r['structure_id']}")
    for r in partial:
        print(f"  [partial] {r['structure_id']}: {r['pairs_skipped']}/{r['pairs_total']} pairs left at zero")


def precache_dataset(df_path, prepared_dir, data_root=None, col_prot_name='PDB',
                      col_prot_chain='Protein chains', col_na_chain='RNA chains',
                      col_label='△G(kcal/mol)', inna_weights=None, inna_repo_path=None,
                      atom_resolution='backbone', device='cpu', **kwargs):
    """Walk the whole master CSV once (fold-agnostic — the same prepared
    files are reused by every fold/split) and write one ComplexData file per
    structure under `prepared_dir`, skipping structures already present.
    Prints a summary of zero-energy fallbacks and returns it as a list of
    dicts (structure_id, status, pairs_total, pairs_skipped), one per
    structure processed in this run (not those already cached)."""
    os.makedirs(prepared_dir, exist_ok=True)
    df = pd.read_csv(df_path)

    inna_model = None
    if inna_weights is not None:
        inna_model = load_inna_model(inna_weights, inna_repo_path, device=device)

    report, n_cached = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        structure_id = row[col_prot_name]
        out_path = os.path.join(prepared_dir, f'{structure_id}.pkl')
        if os.path.exists(out_path):
            n_cached += 1
            continue

        prot_chains = row[col_prot_chain].split(',')
        na_chains = row[col_na_chain].split(',')
        energy = float(row[col_label])

        data, energy_info = prepare_complex(
            data_root,
            cplx_id=structure_id, prot_chains=prot_chains, na_chains=na_chains, energy=energy,
            inna_model=inna_model, atom_resolution=atom_resolution, device=device
        )
        if data is None:
            print(f'[WARN] Skipping {structure_id}: failed to parse structure')
            report.append({'structure_id': structure_id, 'status': 'parse_failed',
                           'pairs_total': 0, 'pairs_skipped': 0})
            continue
        report.append({'structure_id': structure_id, **energy_info})
        data.save(out_path)
    _print_energy_report(report, n_cached)
    return report
