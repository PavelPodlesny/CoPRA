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

def _compute_atom_min_dist(pos_heavyatom, mask_heavyatom):
    L = pos_heavyatom.shape[0]
    distance_map = torch.linalg.norm(
        pos_heavyatom[:, None, :, None, :] - pos_heavyatom[None, :, None, :, :], dim=-1, ord=2
    ).reshape(L, L, -1)
    mask = (mask_heavyatom[:, None, :, None] * mask_heavyatom[None, :, None, :]).reshape(L, L, -1)
    distance_map[~mask] = torch.inf
    return torch.min(distance_map, dim=-1)[0]


def _compute_pairwise_geometry(pos_heavyatom, mask_heavyatom, identifier, seq, atom_resolution='backbone'):
    """Reuses the SelectAtom transform to derive pos_atoms/mask_atoms exactly
    as training would (same atom_resolution), then precomputes the raw
    pairwise backbone-atom distances and phi/psi dihedrals that
    ResiduePairEncoder used to compute on every forward pass."""
    tmp = SelectAtom(resolution=atom_resolution)({
        'pos_heavyatom': pos_heavyatom,
        'mask_heavyatom': mask_heavyatom,
        'identifier': identifier,
        'seq': seq,
    })
    pos_atoms = tmp['pos_atoms']  # (L, A, 3)
    L = pos_atoms.shape[0]

    pos_atoms_b = pos_atoms.unsqueeze(0)  # (1, L, A, 3)
    pairwise_dist = angstrom_to_nm(torch.linalg.norm(
        pos_atoms_b[:, :, None, :, None] - pos_atoms_b[:, None, :, None, :],
        dim=-1, ord=2,
    )).reshape(1, L, L, -1).squeeze(0)  # (L, L, A*A)
    pairwise_dihedral = pairwise_dihedrals(pos_atoms_b).squeeze(0)  # (L, L, 2)
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


# Modern PDB nomenclature -> InNA's (older) RESIDUE_CHARGE_MAP nomenclature,
# for RNA backbone phosphate oxygens.
_ATOM_NAME_ALIASES = {'O1P': 'OP1', 'O2P': 'OP2', 'O3P': 'OP3'}


def compute_interface_energy_map(pdb_path, valid_prot_chains, valid_rna_chains, identifier,
                                  inna_model, device='cpu', batch_size=256):
    """Raw (unembedded) InNA interaction energy for every protein-residue x
    RNA-residue pair in the complex — the same universe of pairs
    ResiduePairEncoder builds features for, no distance cutoff. Same-molecule
    entries and the diagonal are left at 0.0 (InNA has no notion of them).
    Returns (energy_map, energy_mask, info): energy_map and energy_mask are
    (L, L), L = identifier.shape[0]. energy_mask is True only where InNA
    produced a value, so an uncomputed pair (left at 0.0 in energy_map) can be
    told apart from a real energy near 0.
    info = {'status', 'pairs_total', 'pairs_skipped'} records where zeros were
    written as a fallback rather than computed. status is 'ok', 'naskit_failed'
    or 'count_mismatch' (whole map left at zero); pairs_skipped counts
    protein x RNA pairs left at zero because a residue had missing atoms or
    was unknown to InNA."""
    import naskit as nsk
    from model.dataset import ComplexData as InNAComplexData, InNADataset, collate_fn

    L = identifier.shape[0]
    energy_map = torch.zeros(L, L)
    energy_mask = torch.zeros(L, L, dtype=torch.bool)
    info = {'status': 'ok', 'pairs_total': 0, 'pairs_skipped': 0}

    try:
        with nsk.pdbRead(pdb_path) as f:
            pdb = f.read(derive_element=True)[0]
        # Group by chain ID first, then walk `valid_prot_chains`/`valid_rna_chains`
        # in that explicit order — matching `ComplexInput`/`complex_merge`
        # (data/complex.py), which builds `identifier`'s residue order the same
        # way. naskit's own `pdb.prot_chains`/`pdb.na_chains` traversal order is
        # just file order and is *not* guaranteed to match the CSV-specified
        # chain order (e.g. 1JBR: CSV RNA chains "C,D,F" vs file order C,F,D) —
        # iterating pdb.*_chains directly here would silently misalign this
        # energy map against `identifier` for any such structure.
        # Note: naskit splits a single chain ID into multiple discontinuous
        # groups wherever the file has a residue-numbering gap (e.g. 4JYZ's
        # RNA chain 'B' is 8 separate groups) — so this must *accumulate* every
        # group sharing a chain ID, not just keep one, or residues silently
        # go missing.
        prot_chain_groups, na_chain_groups = {}, {}
        for pc in pdb.prot_chains:
            prot_chain_groups.setdefault(pc[0].chain, []).append(pc)
        for nc in pdb.na_chains:
            na_chain_groups.setdefault(nc[0].chain, []).append(nc)
        prot_residues = [res for chain in valid_prot_chains
                          for group in prot_chain_groups.get(chain, []) for res in group]
        rna_residues = [res for chain in valid_rna_chains
                         for group in na_chain_groups.get(chain, []) for res in group]
    except Exception as e:
        print(f'[WARN] naskit failed to parse {pdb_path} ({e}); interface_energy left at zero')
        info['status'] = 'naskit_failed'
        return energy_map, energy_mask, info

    n_prot = int((identifier == 0).sum())
    n_rna = int((identifier == 1).sum())
    if len(prot_residues) != n_prot or len(rna_residues) != n_rna:
        print(f'[WARN] naskit/BioPython residue-count mismatch for {pdb_path} '
              f'(naskit: {len(prot_residues)} prot / {len(rna_residues)} rna, '
              f'BioPython: {n_prot} prot / {n_rna} rna); interface_energy left at zero')
        info['status'] = 'count_mismatch'
        return energy_map, energy_mask, info

    def charge_map_for_residue(r):
        # InNA's RESIDUE_CHARGE_MAP uses older PDB nomenclature for RNA
        # backbone phosphate oxygens (O1P/O2P/O3P); modern PDB files (and
        # naskit's parse of them) use OP1/OP2/OP3 instead. Resolve to
        # whichever name is actually present on this residue.
        base = InNADataset.RESIDUE_CHARGE_MAP.get(r.mname, {})
        resolved = {}
        for old_name, charge in base.items():
            new_name = _ATOM_NAME_ALIASES.get(old_name)
            if new_name is not None and new_name in r:
                resolved[new_name] = charge
            else:
                resolved[old_name] = charge
        return resolved

    def build_pair(prot_res, rna_res):
        natoms = prot_res.natoms + rna_res.natoms
        atoms = torch.zeros((natoms,), dtype=torch.int32)
        charges = torch.zeros((natoms,), dtype=torch.int32)
        i = 0
        try:
            for r in (prot_res, rna_res):
                charge_map = charge_map_for_residue(r)
                for ca in charge_map:
                    if ca not in r:
                        raise ValueError(f'Atom {ca} missing in residue {r.mname}')
                for a in r.atoms():
                    if r.mname not in InNADataset.KNOWN_RESIDUES:
                        raise ValueError(f'Residue {r.mname} unknown to InNA charge table')
                    atoms[i] = InNADataset.ATOM_MAP[a.element]
                    charges[i] = charge_map.get(a.aname, 0)
                    i += 1
        except (KeyError, ValueError):
            return None
        coords = torch.cat([torch.FloatTensor(prot_res.coords), torch.FloatTensor(rna_res.coords)], dim=0)
        roles = torch.cat([torch.ones(prot_res.natoms).int(), 2 * torch.ones(rna_res.natoms).int()], dim=0)
        return InNAComplexData(atoms, charges, roles, coords, None)

    pairs = [(pi, ri, p, r) for pi, p in enumerate(prot_residues) for ri, r in enumerate(rna_residues)]
    info['pairs_total'] = len(pairs)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        items, idx = [], []
        for pi, ri, p, r in chunk:
            item = build_pair(p, r)
            if item is None:
                info['pairs_skipped'] += 1
                continue
            items.append(item)
            idx.append((pi, ri))
        if not items:
            continue
        batch = collate_fn(items).to(device)
        with torch.no_grad():
            energies = inna_model.predict(batch).energy.cpu()
        for (pi, ri), e in zip(idx, energies):
            i, j = pi, n_prot + ri
            energy_map[i, j] = e.item()
            energy_map[j, i] = e.item()
            energy_mask[i, j] = True
            energy_mask[j, i] = True

    return energy_map, energy_mask, info


# ---------------------------------------------------------------------------
# Per-structure preparation and the precache driver
# ---------------------------------------------------------------------------

def prepare_complex(row, data_root, col_prot_name='PDB', col_prot_chain='Protein chains',
                     col_na_chain='RNA chains', col_label='△G(kcal/mol)',
                     inna_model=None, atom_resolution='backbone', device='cpu', **kwargs
                     ) -> Tuple[Optional[ComplexData], Optional[dict]]:
    """Returns (ComplexData, energy_info), or (None, None) if the structure can't be parsed.
    energy_info is compute_interface_energy_map's info dict, with status
    'inna_disabled' when no InNA model was given (all-zero energy map)."""
    structure_id = row[col_prot_name]
    prot_chains = row[col_prot_chain].split(',')
    na_chains = row[col_na_chain].split(',')
    pdb_path = os.path.join(data_root, structure_id + '.pdb')

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
            pdb_path, prot_chains, na_chains, identifier, inna_model, device=device,
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
        structure_id=structure_id,
        label=float(row[col_label]),
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
        data, energy_info = prepare_complex(
            row, data_root, col_prot_name=col_prot_name, col_prot_chain=col_prot_chain,
            col_na_chain=col_na_chain, col_label=col_label, inna_model=inna_model,
            atom_resolution=atom_resolution, device=device,
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
