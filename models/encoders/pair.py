import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import angstrom_to_nm, dihedral_from_four_points
from models.encoders.layers import AngularEncoding
from models.encoders.siren import Siren


def _pairwise_atom_dist(coords_i, coords_j):
    """Raw backbone-atom distances between two residue sets — a rectangular generalization of the square all-pairs case.
    Args:
        coords_i: (N, Li, A, 3)
        coords_j: (N, Lj, A, 3)
    Returns:
        (N, Li, Lj, A*A), in nm
    """
    Li, Lj = coords_i.size(1), coords_j.size(1)
    d = angstrom_to_nm(
        torch.linalg.norm(
            coords_i[:, :, None, :, None] - coords_j[:, None, :, None, :],
            dim=-1, ord=2)
        )
    return d.reshape(coords_i.size(0), Li, Lj, -1)


def _pairwise_dihedrals_block(pos_i, pos_j):
    """Inter-residue Phi/Psi angles between two (possibly different) residue
    sets — a rectangular generalization of `utils.geometry.pairwise_dihedrals`.
    Args:
        pos_i: (N, Li, A, 3)
        pos_j: (N, Lj, A, 3)
    Returns:
        (N, Li, Lj, 2).
    """
    N, Li = pos_i.shape[:2]
    Lj = pos_j.size(1)
    pos_N_i, pos_CA_i, pos_C_i = pos_i[:, :, 0], pos_i[:, :, 1], pos_i[:, :, 2] # (N, Li, 3)
    pos_N_j, pos_CA_j, pos_C_j = pos_j[:, :, 0], pos_j[:, :, 1], pos_j[:, :, 2] # (N, Lj, 3)

    ir_phi = dihedral_from_four_points(
        pos_C_i[:, :, None].expand(N, Li, Lj, 3),
        pos_N_j[:, None, :].expand(N, Li, Lj, 3),
        pos_CA_j[:, None, :].expand(N, Li, Lj, 3),
        pos_C_j[:, None, :].expand(N, Li, Lj, 3),
    )
    ir_psi = dihedral_from_four_points(
        pos_N_i[:, :, None].expand(N, Li, Lj, 3),
        pos_CA_i[:, :, None].expand(N, Li, Lj, 3),
        pos_C_i[:, :, None].expand(N, Li, Lj, 3),
        pos_N_j[:, None, :].expand(N, Li, Lj, 3),
    )
    return torch.stack([ir_phi, ir_psi], dim=-1)


class ResiduePairEncoderBase(nn.Module):
    """Shared embeddings/MLPs for the pairwise structure encoder.
    Subclasses only differ in how they assemble the raw distance/dihedral geometry
    from the precomputed core plus whatever the special pooling
    tokens require. Which one to instantiate is a fixed, config-time (`pooling`) decision.
    """

    # Overridden by subclasses; also used for `interface_energy` padding by
    # the caller (`models/model.py`).
    num_special_tokens = 0

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=30, max_relpos=32, energy_embed_dim=40):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types * self.max_aa_types, max_num_atoms * max_num_atoms)
        # nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms * max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi

        self.energy_embeder = Siren(energy_embed_dim)

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim + energy_embed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def _compute_geometry(self, pairwise_dist, pairwise_dihedral, pos_atoms_special):
        """Returns (d, dihed), both (N, L, L, ...). Implemented by subclasses."""
        raise NotImplementedError

    def forward(self, aa, res_nb, chain_nb, mask_atoms, pairwise_dist, pairwise_dihedral,
                interface_energy, energy_mask, pos_atoms_special=None):
        """
        Args:
            aa, res_nb, chain_nb    : (N, L) — L includes the special tokens
            mask_atoms              : (N, L, 3)
            pairwise_dist           : (N, L - num_special_tokens, L - num_special_tokens, A*A)
            pairwise_dihedral       : (N, L - num_special_tokens, L - num_special_tokens, 2)
            interface_energy        : (N, L, L) raw InNA per-residue-pair energy, already zero-padded for the special tokens
            energy_mask             : (N, L, L) bool, True where interface_energy was actually computed by InNA (padded False for the special tokens)
            pos_atoms_special       : (N, L, A, 3) full backbone coords, including the special tokens coords
        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, 1]  # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # Pair identities
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Relative positions
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L, L)

        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        # Distances/dihedrals
        d, dihed = self._compute_geometry(pairwise_dist, pairwise_dihedral, pos_atoms_special)

        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d ** 2)
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Orientations
        feat_dihed = self.dihedral_embed(dihed)

        # InNA-derived interface energy, embedded via Siren (work-stream B2/B4)
        # Gate by energy_mask: pairs without a computed energy (same-molecule, special tokens, pairs InNA
        # couldn't score) get an all-zero vector, distinct from a real energy near 0, so they add nothing
        # to out_mlp.0 (Siren(0) is otherwise the constant sin(phase)).
        feat_energy = self.energy_embeder(interface_energy) * energy_mask[..., None]  # (N, L, L, energy_embed_dim)

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed, feat_energy], dim=-1)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]
        # if torch.isnan(feat_all).any():
        #     print("Let's check:", torch.isnan(feat_aapair).any(), torch.isnan(feat_relpos).any(), torch.isnan(feat_dist).any(), torch.isnan(feat_dihed).any(), torch.isnan(feat_energy).any())
        #     print("Let's check 2:", torch.isnan(aa).any(), torch.isnan(res_nb).any(), torch.isnan(chain_nb).any(), torch.isnan(mask_atoms).any())
        return feat_all


class PlainResiduePairEncoder(ResiduePairEncoderBase):
    """No pooling tokens (`pooling` != 'token'): `pairwise_dist`/
    `pairwise_dihedral` already cover the full (L, L) sequence as-is."""

    num_special_tokens = 0

    def _compute_geometry(self, pairwise_dist, pairwise_dihedral, pos_atoms_special):
        return pairwise_dist, pairwise_dihedral


class TokenPoolingResiduePairEncoder(ResiduePairEncoderBase):
    """`pooling: token`: 3 synthetic [complex]/[protein]/[rna] rows are
    prepended (s=3) whose positions only exist at forward time (batch/model-
    dependent center of mass), so only the O(s*L) border touching them
    (special<->special, special<->real) is computed fresh; the precomputed
    O((L-s)^2) real<->real block is reused as-is."""

    num_special_tokens = 3

    def _compute_geometry(self, pairwise_dist, pairwise_dihedral, pos_atoms_special):
        assert pos_atoms_special is not None
        s = self.num_special_tokens
        pos_special = pos_atoms_special[:, :s]   # (N, s, A, 3)
        pos_real = pos_atoms_special[:, s:]       # (N, L - s, A, 3)

        d_ss = _pairwise_atom_dist(pos_special, pos_special) # (N, s, s, A*A)
        d_sr = _pairwise_atom_dist(pos_special, pos_real) # (N, s, L-s, A*A)
        d_rs = _pairwise_atom_dist(pos_real, pos_special) # (n, L-s, s, A*A)
        d = torch.cat([
            torch.cat([d_ss, d_sr], dim=2), # (N, s, L, A*A)
            torch.cat([d_rs, pairwise_dist], dim=2), # (N, L-s, L, A*A)
        ], dim=1)  # (N, L, L, A*A)

        dihed_ss = _pairwise_dihedrals_block(pos_special, pos_special)
        dihed_sr = _pairwise_dihedrals_block(pos_special, pos_real)
        dihed_rs = _pairwise_dihedrals_block(pos_real, pos_special)
        dihed = torch.cat([
            torch.cat([dihed_ss, dihed_sr], dim=2),
            torch.cat([dihed_rs, pairwise_dihedral], dim=2),
        ], dim=1)  # (N, L, L, 2)

        return d, dihed
