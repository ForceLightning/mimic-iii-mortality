# Standard Library
from typing import Callable, override

# Third-Party
from rtdl_revisiting_models import MLP

# PyTorch
import torch
from torch import nn

# First party imports
from evid_multimodal_ehr.dst_pytorch import (
    BeliefLayer,
    DempsterLayer,
    DempsterNormalizeLayer,
    DempsterShaferModule,
    DistanceActivationLayer,
    DistanceLayer,
    OmegaLayer,
)

ModuleType = str | Callable[..., nn.Module]
ModuleType0 = str | Callable[[], nn.Module]


def pignistic(mass: torch.Tensor, n_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    # mass: [batch_size, 2, n_class+1]
    probs = mass[:, :n_class] + (1 / n_class) * mass[:, n_class].unsqueeze(1)
    uncertainty = mass[:, n_class]

    return probs, uncertainty


class MassGen(nn.Module):
    def __init__(self, n_feature_maps: int, n_classes: int, n_prototypes: int = 1):
        super().__init__()

        self.ds1 = DistanceLayer(n_prototypes, n_feature_maps)
        self.ds1_activate = DistanceActivationLayer(n_prototypes)
        self.ds2 = BeliefLayer(n_prototypes, n_classes)
        self.ds2_omega = OmegaLayer(n_prototypes, n_classes)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ed = self.ds1(inputs)
        ed_ac = self.ds1_activate(ed)
        mass_prototypes = self.ds2(ed_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        return mass_prototypes_omega


class MLPENNOneStructured(nn.Module):
    def __init__(
        self,
        model: str = "mlp_enn_one_structured",
        pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        prototype_dim: int = 20,
        n_blocks: int = 3,
        structured_d_hidden: int = 32,
        notes_d_hidden: int = 128,
        alpha1: int | float = 2,
        alpha2: int | float = 1,
        dropout: float = 0.1,
        structured_d_in_ls: list[int] | None = None,
        n_class: int = 2,
    ):
        super().__init__()
        self.model = model
        self.pretrained_model = pretrained_model
        self.prototype_dim = prototype_dim
        self.n_blocks = n_blocks
        self.structured_d_hidden = structured_d_hidden
        self.notes_d_hidden = notes_d_hidden
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.dropout = dropout
        self.structured_d_in_ls = (
            [76] if structured_d_in_ls is None else structured_d_in_ls
        )
        self.n_class = n_class

        # Structured
        structured_d_in_ls = self.structured_d_in_ls
        self.backbone = MLP(
            d_in=sum(structured_d_in_ls),
            n_blocks=self.n_blocks,
            d_block=self.structured_d_hidden,
            d_out=self.structured_d_hidden,
            dropout=self.dropout,
        )
        self.structured_cls = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.structured_d_hidden, self.n_class),
        )
        self.structured_dsm = DempsterShaferModule(
            self.structured_d_hidden, self.n_class, self.prototype_dim
        )

        # Notes
        self.notes_reducer = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(768, self.notes_d_hidden)
        )
        self.notes_fcs4logits = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.notes_d_hidden, self.n_class),
        )
        self.notes_dsm = DempsterShaferModule(
            self.notes_d_hidden, self.n_class, self.prototype_dim
        )

        # Fusion
        self.ds_dempster = DempsterLayer(2, self.n_class)
        self.ds_normalize = DempsterNormalizeLayer()

    @override
    def forward(
        self, ehr_data: torch.Tensor, notes_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        structured_feats = self.backbone(ehr_data)  # [bz, t, 32]
        structured_logits = self.structured_cls(structured_feats)  # [bz, t, 2]
        structured_mass = self.structured_dsm(
            structured_feats
        )  # [bz, num_prototypes, 3]

        notes_reduced = self.notes_reducer(notes_data)
        notes_logits = self.notes_fcs4logits(notes_reduced)
        notes_mass = self.notes_dsm(notes_reduced)
        # print(f"ehr_data: {ehr_data.shape}, notes_data: {notes_data.shape}")
        # print(
        #     f"struct_feats: {structured_feats.shape}, notes_reduced: {notes_reduced.shape}"
        # )
        # print(
        #     f"struct_logits: {structured_logits.shape}, notes_logits: {notes_logits.shape}"
        # )
        # print(f"struct_mass: {structured_mass.shape}, notes_mass: {notes_mass.shape}")

        # combine all the mass functions
        mass_ls = [structured_mass, notes_mass]

        # mass_stack: [batch_size, 2, n_class+1] actually [bz, 2, num_prototypes, n_class+1]
        mass_stack = torch.stack(mass_ls, dim=1)
        # print(f"mass_stack: {mass_stack.shape}")
        mass_dempster = self.ds_dempster(mass_stack)
        # print(f"mass_dempster: {mass_dempster.shape}")
        mass_dempster_normalize = self.ds_normalize(mass_dempster)
        # print(f"mass_dempster_normalise: {mass_dempster_normalize.shape}")

        probs, uncertainty = pignistic(mass_dempster_normalize, self.n_class)
        # print(f"probs: {probs.shape}, uncertainty: {uncertainty.shape}")

        return probs, structured_logits, notes_logits, uncertainty
