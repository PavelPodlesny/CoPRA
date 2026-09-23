import torch.nn as nn
import torch
import esm
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.encoders.pair import PlainResiduePairEncoder, TokenPoolingResiduePairEncoder
from models.register import ModelRegister
from models.components.coformer import CoFormer
import torch.nn.functional as F
from models.lora_tune import LoRAESM, LoRARiNALMo, ESMConfig, RiNALMoConfig
from data.complex import SUPER_PROT_IDX, SUPER_RNA_IDX, SUPER_CPLX_IDX, SUPER_CHAIN_IDX

from peft import (
    LoraConfig,
    get_peft_model,
)
R = ModelRegister()

def load_esm(esm_type):
    if esm_type == '650M':
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
    elif esm_type == '150M':
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    elif esm_type == '35M':
        model, _ = esm.pretrained.esm2_t12_35M_UR50D()
    elif esm_type == '8M':
        model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size
    
def load_rinalmo(rinalmo_weights, rinalmo_type):
    if rinalmo_type == '650M':
        size = 'giga'
    elif rinalmo_type == '150M':
        size = 'mega'
    elif rinalmo_type == '35M':
        size = 'micro'
    elif rinalmo_type == '8M':
        size = 'nano'
    config = model_config(size)
    model = RiNALMo(config)
    # alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx):
    # print("Input shape:", prot_embedding.shape, na_embedding.shape)
    # result = prot_embedding.new_full([len(prot_embedding), seq_len, prot_embedding.shape[-1]], 0) # (N, L, E)
    new_complexes = []
    masks = []
    for i in range(len(prot_embedding)):
        item_prot_embed = prot_embedding[i]
        item_prot_mask = prot_mask[i]
        item_na_embed = na_embedding[i]
        item_na_mask = na_mask[i]
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad)
    result = torch.stack(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks

def segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx=None):
    cum_prot = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(prot_chains), dim=0)]).int()
    cum_na = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(na_chains), dim=0)]).int()
    new_complexes = []
    masks = []
    for i, (s_prot, e_prot, s_na, e_na) in enumerate(zip(cum_prot[:-1], cum_prot[1:], cum_na[:-1], cum_na[1:])):
        item_prot_embed = prot_embedding[s_prot:e_prot].reshape((-1, prot_embedding.shape[-1]))
        item_prot_mask = prot_mask[s_prot:e_prot].reshape(-1)
        item_na_embed = na_embedding[s_na: e_na].reshape((-1, na_embedding.shape[-1]))
        item_na_mask = na_mask[s_na: e_na].reshape(-1)
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        # # selected_pad = torch.cat([selected, torch.zeros((seq_len-len(selected), prot_embedding.shape[-1]), device=selected.device)], dim=0)
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))
    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    # print("Result shape:", result)
    return result, masks

@R.register('copra')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 rinalmo_type='650M',
                 pooling='mean',
                 output_dim=1,
                 pair_dim=320,
                 fix_lms=True,
                 lora_tune=True,
                 lora_rank=16,
                 lora_alpha=32,
                 representation_layer=33,
                 energy_embed_dim=40,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.esm, esm_feat_size = load_esm(esm_type)
        self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights, rinalmo_type)
        pair_encoder_cls = TokenPoolingResiduePairEncoder if pooling == 'token' else PlainResiduePairEncoder
        self.pair_encoder = pair_encoder_cls(pair_dim, max_num_atoms=4, energy_embed_dim=energy_embed_dim)  # N, CA, C, O,
        self.c_former = CoFormer(**kwargs['coformer'])
        self.representation_layer = representation_layer
        self.proj = 0
        if esm_feat_size != rinalmo_feat_size:
            self.proj = 1
            self.project_feat= nn.Linear(esm_feat_size, rinalmo_feat_size)
        self.complex_dim = kwargs['coformer']['embed_dim']
        self.feat_size = rinalmo_feat_size
        self.proj_cplx= nn.Linear(self.feat_size, self.complex_dim)
        if lora_tune:
            import re
            pattern = r'\((\w+)\): Linear'
            rinalmo_linear_layers = re.findall(pattern, str(self.rinalmo.modules))
            rinalmo_linear_modules = list(set(rinalmo_linear_layers))
            print("In rinalmo:", rinalmo_linear_modules)
            rinalmo_linear_modules = ['Wqkv']
            esm_linear_layers = re.findall(pattern, str(self.esm.modules))
            esm_linear_modules = list(set(esm_linear_layers))
            print("In esm:", esm_linear_modules)
            print("Getting Lora Models...")
            # copied from LongLoRA
            rinalmo_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=rinalmo_linear_modules,
                lora_alpha=lora_alpha
            )
            esm_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=esm_linear_modules,
                lora_alpha=lora_alpha
            )
            rinalmo_config = RiNALMoConfig()
            esm_config = ESMConfig()
            self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
            self.esm = LoRAESM(self.esm, esm_config)
            self.rinalmo = get_peft_model(self.rinalmo, rinalmo_lora_config)
            print("Get RINALMO DONE!!!!!")
            self.rinalmo.print_trainable_parameters()
            self.esm = get_peft_model(self.esm, esm_lora_config)
            print("Get ESM DONE!!!!!")
            self.esm.print_trainable_parameters()
        elif fix_lms:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
            for p in self.esm.parameters():
                p.requires_grad_(False)
                
        self.pooling = pooling
        print("Pooling Strategy:", self.pooling)
        if self.pooling == 'token':
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)
        if pair_dim != self.complex_dim:
            self.z_proj = nn.Linear(pair_dim, self.complex_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )

    def _forward(self, input, strategy='separate'):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']

        with torch.cuda.amp.autocast():
            prot_embedding = self.esm(prot_input, repr_layers=[self.representation_layer], return_contacts=False)['representations'][self.representation_layer]
            na_embedding = self.rinalmo(na_input)['representation']
            if self.proj:
                prot_embedding = self.project_feat(prot_embedding)

        prot_embedding = prot_embedding.float()
        na_embedding = na_embedding.float()
        max_len = input['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer, i.e. padding
        if 'patch_idx' in input:
            patch_idx = input['patch_idx']
        else:
            patch_idx = None
        if strategy == 'separate':
            # input shape [N', L], where N' is flexible in every batch
            out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']
        else:
            out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks # True ~ 1 ~ mask/ignore the position 
        
        aa=input['restype']
        res_nb=input['res_nb']
        chain_nb=input['chain_nb']
        pos_atoms=input['pos_atoms']
        mask_atoms=input['mask_atoms']
        pairwise_dist=input['pairwise_dist']
        pairwise_dihedral=input['pairwise_dihedral']
        interface_energy=input['interface_energy']

        if self.pooling == 'token':
            mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
            cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
            prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
            rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
            
            out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
            key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
            
            cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
            prot_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_PROT_IDX
            rna_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_RNA_IDX
            aa = torch.cat([cplx_type, prot_type, rna_type, aa], dim=1)
            
            res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
            res_nb_prot = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
            res_nb_rna = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
            
            res_nb = torch.cat([res_nb_cplx, res_nb_prot, res_nb_rna, res_nb], dim=1)
            super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
            chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)
            
            center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
            center_prot = ((pos_atoms * (1-input['identifier'])[:, :, None, None] * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((1-input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            center_rna = ((pos_atoms * (input['identifier'][:, :, None, None]) * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            pos_atoms = torch.cat([center_cplx, center_prot, center_rna, pos_atoms], dim=1)
            # noise = torch.randn_like(pos_atoms, dtype=torch.float32, device=pos_atoms.device)
            # pos_atoms += noise
            mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
            mask_atom[:,:,0] = 1
            mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)

        num_special_tokens = self.pair_encoder.num_special_tokens
        if num_special_tokens > 0:
            interface_energy = F.pad(interface_energy, (num_special_tokens, 0, num_special_tokens, 0))

        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            mask_atoms=mask_atoms,
            pairwise_dist=pairwise_dist,
            pairwise_dihedral=pairwise_dihedral,
            interface_energy=interface_energy,
            pos_atoms_special=pos_atoms if num_special_tokens > 0 else None,
        )

        return out_embedding, z, key_padding_mask

    def forward(self, input, strategy='separate'):
        out_embedding, z, key_padding_mask = self._forward(input, strategy)

        output, z, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)

        complex_embedding = output + self.z_proj(z).sum(-2) * 0.001
        if self.pooling == 'token':
            complex_embedding = output[:, 0, :].squeeze(1)
        else:
            complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
            if self.pooling == 'mean':
                # Prot_mask: [N, L]
                seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                complex_embedding = complex_embedding / (seq_mask_sum + 1e-10)

        output = self.pred_head(complex_embedding)
        output = output.squeeze(1)
        return output
