from typing import override
import torch
from torch import nn
import numpy as np


class DistanceLayer(nn.Module):
    def __init__(self, n_prototypes: int, n_feature_maps: int):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(n_prototypes, n_feature_maps))
        nn.init.normal_(self.w)
        self.n_prototypes = n_prototypes

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_prototypes):
            if i == 0:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = un_mass_i

            if i >= 1:
                un_mass_i = (self.w[i, :] - inputs) ** 2
                un_mass_i = torch.sum(un_mass_i, dim=-1, keepdim=True)
                un_mass = torch.cat([un_mass, un_mass_i], -1)  # pyright: ignore
        return un_mass  # pyright: ignore


class DistanceActivationLayer(nn.Module):
    def __init__(
        self,
        n_prototypes: int,
        init_alpha: int | float = 0,
        init_gamma: int | float = 0.1,
    ):
        super().__init__()
        self.eta = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        self.xi = nn.Linear(in_features=n_prototypes, out_features=1, bias=False)
        nn.init.constant_(self.eta.weight, init_gamma)
        nn.init.constant_(self.xi.weight, init_alpha)
        self.n_prototypes = n_prototypes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gamma = torch.square(self.eta.weight)
        alpha = torch.neg(self.xi.weight)
        alpha = torch.exp(alpha) + 1
        alpha = torch.div(1, alpha)

        si = torch.mul(gamma, inputs)
        si = torch.neg(si)
        si = torch.exp(si)
        si = torch.mul(si, alpha)
        max_val, _max_idx = torch.max(si, dim=-1, keepdim=True)
        si /= max_val + 0.0001

        return si


class BeliefLayer(nn.Module):
    def __init__(self, n_prototypes: int, num_class: int):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.Tensor(n_prototypes, num_class))
        torch.nn.init.normal_(self.beta)

        self.prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        beta = torch.square(self.beta)
        beta_sum = torch.sum(beta, dim=-1, keepdim=True)
        u = torch.div(beta, beta_sum)
        inputs_new = torch.unsqueeze(inputs, dim=-1)
        for i in range(self.prototypes):
            if i == 0:
                # batch_size * n_class
                mass_prototype_i = torch.mul(u[i, :], inputs_new[:, i])
                mass_prototype = torch.unsqueeze(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = torch.unsqueeze(
                    torch.mul(u[i, :], inputs_new[:, i]), -2
                )
                mass_prototype = torch.cat(
                    [mass_prototype, mass_prototype_i], -2  # pyright: ignore
                )
        return mass_prototype  # pyright: ignore


class OmegaLayer(nn.Module):
    def __init__(self, n_prototypes: int, num_class: int) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mass_omega_sum = 1 - torch.sum(inputs, -1, keepdim=True)
        mass_with_omega = torch.cat([inputs, mass_omega_sum], -1)
        return mass_with_omega


class DempsterLayer(nn.Module):
    def __init__(self, n_prototypes: int, num_class: int) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch_size, features, n_class]
        # output: [batch_size, n_prototypes, n_class]
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class DempsterNormalizeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        return mass_combine_normalize


class DempsterShaferModule(nn.Module):
    def __init__(self, n_feature_maps: int, n_classes: int, n_prototypes: int = 1):
        super().__init__()

        self.ds1 = DistanceLayer(
            n_prototypes=n_prototypes, n_feature_maps=n_feature_maps
        )
        self.ds1_activate = DistanceActivationLayer(n_prototypes=n_prototypes)
        self.ds2 = BeliefLayer(n_prototypes=n_prototypes, num_class=n_classes)
        self.ds2_omega = OmegaLayer(n_prototypes=n_prototypes, num_class=n_classes)

        self.ds3_dempster = DempsterLayer(
            n_prototypes=n_prototypes, num_class=n_classes
        )
        self.ds3_normalize = DempsterNormalizeLayer()

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ed = self.ds1(inputs)
        ed_ac = self.ds1_activate(ed)
        mass_prototypes = self.ds2(ed_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_dempster = self.ds3_dempster(mass_prototypes_omega)
        mass_dempster_normalize = self.ds3_normalize(mass_dempster)
        return mass_dempster_normalize


def tile(a: torch.Tensor, dim: int, n_tile: int, device: torch.device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    ).to(device)
    return torch.index_select(a, dim, order_index)


class DM(nn.Module):
    def __init__(
        self,
        num_class: int,
        nu: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.nu = nu
        self.num_class = num_class
        self.device = device

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upper = torch.unsqueeze(
            (1 - self.nu) * inputs[..., -1], -1
        )  # here 0.1 = 1 - \nu
        upper = tile(upper, dim=-1, n_tile=self.num_class + 1, device=self.device)
        outputs = (inputs + upper)[..., :-1]
        return outputs
