# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
# https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py
from typing import Tuple, Union, Optional, List

from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pickle
from utils.patch_selector import (PatchImportanceTracker, RandomPatchSelector, MetaLearnerPatchSelector,
                                  SVMPatchSelector, RandomForestPatchSelector, KNNPatchSelector)


class CutLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size
    
    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        res = int(np.sqrt(res2))

        ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2) 
        trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss


class EnhancedCutLoss(CutLoss):
    """ A samrt cut loss class using different patch selection methods """
    def __init__(self, device, n_patches=256, patch_size=1, selector_type='random', feature_dim=1280):
        super().__init__(n_patches, patch_size)
        self.n_patches = n_patches
        self.losses = []
        self.accuracies = []

        # Initialize patch selector based on type
        if selector_type == 'random':
            self.patch_selector = RandomPatchSelector()
        elif selector_type == 'meta':
            self.patch_selector = MetaLearnerPatchSelector(device, feature_dim)
        elif selector_type == 'svm':
            self.patch_selector = SVMPatchSelector(feature_dim)
        elif selector_type == 'knn':
            self.patch_selector = KNNPatchSelector(feature_dim)
        elif selector_type == 'rf':
            self.patch_selector = RandomForestPatchSelector(feature_dim)
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

        self.importance_tracker = PatchImportanceTracker(feature_dim)

    def train_selector(self):
        """Train the patch selector using collected importance data"""
        if not isinstance(self.patch_selector, RandomPatchSelector):
            features, labels = self.importance_tracker.get_training_data()
            loss = self.patch_selector.train_step(features, labels)
            accuracy = self.patch_selector.get_accuracy(features, labels)
            self.losses.append(loss)
            self.accuracies.append(accuracy)

    def plot_losses_accuracies(self):
        if not isinstance(self.patch_selector, RandomPatchSelector):
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, c='r')
            plt.title("Patch selector training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss value")
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(self.accuracies, c='b')
            plt.title("Patch selector training accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy %")
            plt.grid()
            plt.show()

            with open("results/selector/accuracies", "wb") as fp:
                pickle.dump(self.accuracies, fp)

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        res = int(np.sqrt(res2))

        ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)
        trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)

            # Use patch selector instead of random permutation
            patch_features = ref_noise_pooled[:1].flatten(0, 1)
            patch_ids = self.patch_selector.select_patches(
                patch_features,
                min(self.n_patches, ref_noise_pooled.shape[1])
            )
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            trg_sample = trg_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            loss += self.PatchNCELoss(ref_sample, trg_sample).mean()

            if not isinstance(self.patch_selector, RandomPatchSelector):
                patch_loss = torch.norm(ref_noise_pooled - trg_noise_pooled, dim=-1)
                self.importance_tracker.update(patch_features, patch_loss)

        return loss


class FrameworkLoss(ABC):
    """ Abstract class for framework loss used by different methods"""
    @abstractmethod
    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        pass

    @abstractmethod
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):
        pass

    @abstractmethod
    def process_epsilon_prediction(self, eps_pred, z_trg, z_t_trg, timestep, timestep_prev):
        pass


class DDSLoss(FrameworkLoss):

    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )
        timestep_prev = timestep

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep, timestep_prev
    
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(latent_input, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def process_epsilon_prediction(self, eps_pred, z_trg, z_t_trg, timestep, timestep_prev):
        return eps_pred

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device


class PDSLoss(DDSLoss):

    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        self.scheduler.set_timesteps(1000)
        timesteps = reversed(self.scheduler.timesteps)

        if timestep is None:
            b = z.shape[0]
            idx = torch.randint(
                low = self.t_min,
                high = self.t_max,
                size=(b,),
                device="cpu",
                dtype=torch.long
            )
            timestep = timesteps[idx].to(self.device)
            timestep_prev = timesteps[idx - 1].to(self.device)
        else:
            timestep_prev = timestep

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)

        return z_t, eps, timestep, timestep_prev

    def compute_posterior_mean(self, xt, noise_pred, t, t_prev):
        """
        Computes an estimated posterior mean \mu_\phi(x_t, y; \epsilon_\phi).
        """
        device = self.device
        beta_t = self.scheduler.betas[t.to("cpu")].to(device)
        alpha_t = self.scheduler.alphas[t.to("cpu")].to(device)
        alpha_bar_t = self.scheduler.alphas_cumprod[t.to("cpu")].to(device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[t_prev.to("cpu")].to(device)

        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(
            alpha_bar_t
        )
        c0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        c1 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        mean_func = c0 * pred_x0 + c1 * xt
        return mean_func

    def process_epsilon_prediction(self, eps_pred, z_trg, z_t_trg, timestep, timestep_prev):
        noise_prev = torch.randn_like(z_trg)
        x_t_prev = self.scheduler.add_noise(z_trg, noise_prev, timestep_prev)
        beta_t = self.scheduler.betas[timestep.to("cpu")].to(self.device)
        alpha_bar_t = self.scheduler.alphas_cumprod[timestep.to("cpu")].to(self.device)
        alpha_bar_t_prev = self.scheduler.alphas_cumprod[timestep_prev.to("cpu")].to(self.device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)
        mu = self.compute_posterior_mean(z_t_trg, eps_pred, timestep, timestep_prev)
        zt = (x_t_prev - mu) / sigma_t

        return zt