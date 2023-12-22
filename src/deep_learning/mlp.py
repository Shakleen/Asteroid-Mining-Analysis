from typing import List

import torch
from torch.nn import (
    LazyLinear,
    Dropout,
    ReLU,
    Sequential,
    LayerNorm,
    MSELoss,
)
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(29)


class SimpleBlock(torch.nn.Module):
    def __init__(self, num_output: int, dropout: float, device: str = "cuda"):
        super().__init__()

        self.net = Sequential(
            LazyLinear(num_output, device=device),
            ReLU(),
            Dropout(dropout),
        )

    def __call__(self, X):
        return self.net(X)


class Block(torch.nn.Module):
    def __init__(
        self,
        num_output_list: List[int],
        dropout_list: List[float],
        num_output: int,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.hidden_net = Sequential(
            *[
                SimpleBlock(no, d, device)
                for no, d in zip(num_output_list, dropout_list)
            ]
        )

        self.layer_out = LazyLinear(num_output, device=device)

    def __call__(self, X):
        return self.layer_out(self.hidden_net(X))


class MLP(torch.nn.Module):
    def __init__(
        self,
        n: int,
        num_output_list: List[int],
        dropout_list: List[int],
        block_io_shape: int = 64,
        device: str = "cuda",
    ):
        super().__init__()

        self.layer_in = LazyLinear(block_io_shape, device=device)

        self.block_list = [
            Block(
                num_output_list=num_output_list,
                dropout_list=dropout_list,
                num_output=block_io_shape,
            )
            for _ in range(n)
        ]
        self.norm_list = [LayerNorm(block_io_shape, device=device) for _ in range(n)]
        self.output_layer = LazyLinear(1, device=device)
        self.loss_func = MSELoss()
        self.apply(self._init)

    def _init(self, module):
        if type(module) is torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def __call__(self, X):
        output = self.layer_in(X)

        for layer, norm in zip(self.block_list, self.norm_list):
            output = F.relu(norm(layer(output) + output))

        return self.output_layer(output)

    def loss(self, y, preds):
        raise NotImplementedError


class MLP_Diameter(MLP):
    def loss(self, y, preds):
        base_loss = self.loss_func(preds, y)
        
        e = torch.tensor(1e8, device=device)**preds

        # Penalize values less than 0.0
        denominator = torch.min(
            torch.min(e),
            torch.tensor(1.0, device=device),
        )

        return base_loss / denominator


class MLP_Albedo(MLP):
    def loss(self, y, preds):
        base_loss = self.loss_func(preds, y)

        eb = torch.tensor(1e8, device=device)
        e = eb**preds

        # Penalize values greater than 1.0
        numerator = torch.max(torch.max(e), eb) / eb

        # Penalize values less than 0.0
        denominator = torch.min(
            torch.min(e),
            torch.tensor(1.0, device=device),
        )

        return (base_loss * numerator) / denominator
