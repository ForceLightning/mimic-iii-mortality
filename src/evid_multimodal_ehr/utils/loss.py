# Standard Library
from typing import override

# PyTorch
import torch
from torch import nn


class MultitaskLoss4OneStructured(nn.modules.loss._Loss):
    def __init__(
        self,
        alpha1: int | float,
        alpha2: int | float,
        weight: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.nll_loss = torch.nn.NLLLoss(weight=weight)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eps = 1e-10

    @override
    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        main_probs, structured_logits, notes_logits = inputs
        main_loss = self.nll_loss(torch.log(main_probs), targets.squeeze(1))
        aux_loss_1 = self.ce_loss(structured_logits, targets.squeeze(1))
        aux_loss_2 = self.ce_loss(notes_logits, targets.squeeze(1))

        # return (
        #     main_loss / (main_loss.detach() + self.eps)
        #     + self.alpha1 * aux_loss_1 / (aux_loss_1.detach() + self.eps)
        #     + self.alpha2 * aux_loss_2 / (aux_loss_2.detach() + self.eps)
        # )
        return main_loss + self.alpha1 * aux_loss_1 + self.alpha2 * aux_loss_2


class MultitaskLoss4OneStructured_detached(nn.modules.loss._Loss):
    def __init__(
        self,
        alpha1: int | float,
        alpha2: int | float,
        weight: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.nll_loss = torch.nn.NLLLoss(weight=weight)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eps = 1e-10

    @override
    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        main_probs, structured_logits, notes_logits = inputs
        main_loss = self.nll_loss(torch.log(main_probs), targets.squeeze(1))
        aux_loss_1 = self.ce_loss(structured_logits, targets.squeeze(1))
        aux_loss_2 = self.ce_loss(notes_logits, targets.squeeze(1))

        return (
            main_loss + self.alpha1 * aux_loss_1 + self.alpha2 * aux_loss_2
        ).detach()
