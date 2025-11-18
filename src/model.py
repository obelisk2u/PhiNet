from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTPP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        hazard_hidden_dim: int = 32,
        init_hidden_scale: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        # GRU history encoder: input is scalar Δt
        self.gru_cell = nn.GRUCell(input_size=1, hidden_size=hidden_dim)

        # Cumulative hazard network Φ(τ | h)
        # input: [τ, h] -> hidden -> scalar
        self.hazard_mlp = nn.Sequential(
            nn.Linear(1 + hidden_dim, hazard_hidden_dim),
            nn.Tanh(),
            nn.Linear(hazard_hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

        if init_hidden_scale > 0:
            self.h0 = nn.Parameter(init_hidden_scale * torch.randn(hidden_dim))
        else:
            self.h0 = None

        self.to(device)

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        if self.h0 is None:
            return torch.zeros(batch_size, self.hidden_dim, device=self.device)
        else:
            return self.h0.unsqueeze(0).expand(batch_size, -1)

    def cumulative_hazard(self, tau: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tau.unsqueeze(-1), h], dim=-1)   
        phi_raw = self.hazard_mlp(x).squeeze(-1)        
        phi = self.softplus(phi_raw)                    
        return phi

    def intensity(
        #compute delphi/deltau and lambda with autodiff
        self, tau: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau_var = tau.detach().clone().requires_grad_(True)

        phi = self.cumulative_hazard(tau_var, h) 

        grad_phi = torch.autograd.grad(
            phi.sum(), tau_var, create_graph=True
        )[0] 

        lambda_pos = F.softplus(grad_phi) + 1e-8

        return lambda_pos, phi

    def masked_sequence_log_likelihood(
        self,
        deltas: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        #loglik calculation
        B, L = deltas.shape
        h = self.initial_hidden(B)  

        total_loglik = 0.0
        total_events = mask.sum()  

        for t_idx in range(L):
            tau_i = deltas[:, t_idx]    
            mask_i = mask[:, t_idx]    

            if mask_i.sum() == 0:
                continue

            lambda_i, phi_i = self.intensity(tau_i, h)  

            log_lambda_i = torch.log(lambda_i)
            loglik_i = log_lambda_i - phi_i    
 
            total_loglik = total_loglik + (loglik_i * mask_i).sum()
 
            new_h = self.gru_cell(tau_i.unsqueeze(-1), h)   
            mask_i_exp = mask_i.unsqueeze(-1)               
            h = mask_i_exp * new_h + (1.0 - mask_i_exp) * h

        avg_loglik = total_loglik / (total_events + 1e-8)
        return avg_loglik

    def forward(self, deltas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.masked_sequence_log_likelihood(deltas, mask)