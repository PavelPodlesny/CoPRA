import torch

from ._base import register_transform, _index_select_complex


@register_transform('selected_region_with_distmap')
class SelectedRegionWithDistmap(object):

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, data):
        atoms_dist_min = data['atom_min_dist']

        identifier = data['identifier']
        tmp = atoms_dist_min[identifier==0]
        interface_distance = tmp[:, identifier==1]
        prot_min_dist = interface_distance.min(dim=1)[0]
        rna_min_dist = interface_distance.transpose(0, 1).min(dim=1)[0]
        total_min = torch.cat([prot_min_dist, rna_min_dist], dim=0)
        patch_idx = torch.argsort(total_min)[:self.patch_size]
        patch_idx, _ = torch.sort(patch_idx)
        # print(self.patch_size)
        data_patch = _index_select_complex(data, patch_idx)
        data_patch['patch_idx'] = patch_idx
        return data_patch
