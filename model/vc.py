import torch
from model.base import BaseModule
from model.diffusion import Diffusion
from model.utils import sequence_mask, fix_len_compatibility

class DiffVC_Plus(BaseModule):
    def __init__(self, n_feats, 
                 spk_dim, dec_dim, 
                 beta_min, beta_max):
        super().__init__()
        self.n_feats = n_feats
        self.spk_dim = spk_dim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, 
                                 beta_min, beta_max)

 
    @torch.no_grad()
    def forward(self, x, x_lengths, c, unit, n_timesteps, 
                mode='ml'):

        x, x_lengths = self.relocate_input([x, x_lengths])
        c = self.relocate_input(c)
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = unit
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, 1.0)  

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, 768, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, c, 
                         n_timesteps, mode) 
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, c, unit):
        x, x_lengths, c = self.relocate_input([x, x_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        diff_loss = self.decoder.compute_loss(x, x_mask, unit, c)
        return diff_loss

