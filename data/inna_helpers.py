"""Converts ComplexInput.from_path's output into a list of InNA ComplexData points, one per
(protein residue, rna residue) pair.

Precondition: the InNA repo is already on sys.path (as `load_inna_model` does for the rest of
data/prepare.py), so `from model.dataset import ComplexData, InNADataset` resolves.
"""
import numpy as np
import torch

from data.protein.residue_constants import (
    restypes_with_x, restype_1to3_with_unk, restype_name_to_atom14_names,
)
from data.rna.base_constants import RNA_ATOMS, NUM_TO_LETTER
from model.dataset import ComplexData as InNAComplexData, InNADataset

# RESIDUE_CHARGE_MAP uses the older O1P/O2P/O3P names for RNA backbone phosphate oxygens;
# RNA_ATOMS (and therefore the atom names this function looks up) uses OP1/OP2/OP3.
_ATOM_NAME_ALIASES = {'O1P': 'OP1', 'O2P': 'OP2', 'O3P': 'OP3'}

_ATOM_MAP = InNADataset.ATOM_MAP  # {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'P':15, 'S':16}


def _element_of(atom_name):
    """First letter of a heavy-atom PDB name is its element for every name that appears in
    restype_name_to_atom14_names / RNA_ATOMS -- no two-letter-element atoms (metals, halogens)
    occur on standard amino acid / nucleotide backbones+sidechains."""
    return atom_name[0]


def _residue_charge_map(resname):
    """resname -> {atom_name: charge}, atom names in the OP1/OP2/OP3 convention (RNA_ATOMS),
    not RESIDUE_CHARGE_MAP's own O1P/O2P/O3P convention."""
    atom2charge = InNADataset.RESIDUE_CHARGE_MAP.get(resname, {})
    return {_ATOM_NAME_ALIASES.get(atom_name, atom_name): charge for atom_name, charge in atom2charge.items()}


def _residue_atom_tensors(cplx):
    """Per-residue InNA atom tensors (element code, charge, coords), computed once per residue
    rather than once per pair -- each residue participates in many pairs (every protein residue
    pairs with every rna residue), so amortizing this away from the O(n_prot * n_rna) pairing
    loop is a real saving, not just style.

    Returns two dicts {row_index: (atoms, charges, coords)}, keyed by row index into
    cplx.restype/identifier -- one for protein residues (identifier==0), one for rna residues
    (identifier==1). A residue InNA can't identify (UNK restype, RNA restype outside A/G/C/U, a
    resname not in KNOWN_RESIDUES) or whose every atom is unresolved is simply absent from its
    dict, so it never appears in any pair.
    """
    protein, rna = {}, {}
    L = cplx.restype.shape[0]
    for i in range(L):
        restype = int(cplx.restype[i])
        is_rna = bool(cplx.identifier[i])

        if is_rna:
            resname = restype - 21
            if not (0 <= resname < len(NUM_TO_LETTER)):
                continue  # restype outside A/G/C/U -> unscorable
            resname = NUM_TO_LETTER[resname]  # 'A' / 'G' / 'C' / 'U'
            slot_names = RNA_ATOMS             # same 27-name list for every base
            atom41_offset = 14                 # RNA occupies atom41[:, 14:41]
        else:
            if not (0 <= restype < len(restypes_with_x)):
                continue  # out-of-range restype (defensive; shouldn't happen pre-padding)
            resname = restype_1to3_with_unk(restype)  # 'ASP' / ... / 'UNK'
            slot_names = restype_name_to_atom14_names[resname]
            atom41_offset = 0                  # protein occupies atom41[:, :14]

        if resname not in InNADataset.KNOWN_RESIDUES:
            continue  # UNK, or a modified residue InNA has never seen

        charge_map = _residue_charge_map(resname)
        # slot_of = {name: slot for slot, name in enumerate(slot_names) if name}

        # InNA's own get_charge/build_pair treat a residue as ENTIRELY unscorable if it is
        # missing an atom that carries a nonzero charge for its residue type (e.g. ASP missing
        # OD1/OD2) -- not score it with the rest of its atoms as if that charge were absent.
        # Check this before collecting atoms, not just "skip the missing atom".
        # if any(cplx.atom41_mask[i, atom41_offset + slot_of[name]] == 0 if name in slot_of else True
        #        for name in charge_map):
        #     continue  # missing (or nonexistent) a charge-bearing atom -> unscorable

        atoms_i, charges_i, coords_i = [], [], []
        for slot, name in enumerate(slot_names):
            if not name: # empty string
                continue  # unused atom14 slot for this residue type
            atom41_idx = atom41_offset + slot
            if not cplx.atom41_mask[i, atom41_idx]: # False ~ unresolved
                continue  # real atom but unresolved in this structure
            atoms_i.append(_ATOM_MAP[_element_of(name)])
            charges_i.append(charge_map.get(name, 0))
            coords_i.append(cplx.atom41_positions[i, atom41_idx])

        if not atoms_i:
            continue  # recognized residue, but every one of its atoms is unresolved -> unscorable

        entry = (
            torch.tensor(atoms_i, dtype=torch.int32),
            torch.tensor(charges_i, dtype=torch.int32),
            torch.tensor(np.stack(coords_i, axis=0), dtype=torch.float32),
        )
        if is_rna:
            rna[i] = entry
        else:
            protein[i] = entry

    return protein, rna


def complex_to_inna_pairs(cplx):
    """Every (protein residue, rna residue) pair in `cplx`, each as an InNA ComplexData ready for
    `collate_fn`.

    Returns:
        pairs:   List[ComplexData], energy=None. Each item's atoms/charges/coords are
                 [that pair's protein-residue atoms; its rna-residue atoms] (protein first,
                 matching the current build_pair convention), roles = [1]*n_prot_atoms +
                 [2]*n_rna_atoms.
        indices: List[(i, j)], same length and order as `pairs` -- i, j are row indices into
                 cplx.restype/identifier, i.e. `pairs[k]` is exactly the InNA input for
                 `energy_map[i, j]` (and its mirror `energy_map[j, i]`).

        Both lists are in a fixed order: for each kept protein residue index i (ascending), for
        each kept rna residue index j (ascending) -- so `indices` is sorted lexicographically.
        A residue InNA can't score (see `_residue_atom_tensors`) contributes no pairs at all:
        every (i, *) or (*, j) touching it is simply absent from both lists, not zero-filled --
        the caller is responsible for leaving those `energy_map`/`energy_mask` cells
        at their default.
    """
    protein, rna = _residue_atom_tensors(cplx)
    pairs, indices = [], []
    for i in sorted(protein):
        p_atoms, p_charges, p_coords = protein[i]
        p_roles = torch.ones(len(p_atoms), dtype=torch.int32)
        for j in sorted(rna):
            r_atoms, r_charges, r_coords = rna[j]
            r_roles = 2 * torch.ones(len(r_atoms), dtype=torch.int32)
            pairs.append(InNAComplexData(
                atoms=torch.cat([p_atoms, r_atoms]),
                charges=torch.cat([p_charges, r_charges]),
                roles=torch.cat([p_roles, r_roles]),
                coords=torch.cat([p_coords, r_coords], dim=0),
                energy=None,
            ))
            indices.append((i, j))
    return pairs, indices