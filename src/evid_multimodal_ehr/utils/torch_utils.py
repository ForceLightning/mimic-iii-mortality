# Third-Party
from rich.console import Console
from rich.table import Table

# PyTorch
from torch import nn

CONSOLE = Console()


def count_parameters(model: nn.Module) -> int:
    total_params = 0
    table = Table(title="Model parameters")
    table.add_column("Modules")
    table.add_column("Parameters")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
        else:
            params = 0
        table.add_row(name, f"{params}")
        total_params += params

    CONSOLE.print(table)
    CONSOLE.print(f"* Total number of trainable params: {total_params}")
    return total_params
