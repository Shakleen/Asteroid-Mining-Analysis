from typing import List

import torch
from torch.nn import LazyLinear, Dropout, ReLU, Sequential, LayerNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(29)


class SimpleBlock(torch.nn.Module):
    def __init__(self, num_output: int, dropout: float, device: str = "cuda"):
        super().__init__()

        self.net = Sequential(
            LazyLinear(num_output, device=device),
            LayerNorm(num_output, device=device),
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

        self.layers = [
            SimpleBlock(no, d, device) for no, d in zip(num_output_list, dropout_list)
        ]

        self.layer_out = LazyLinear(num_output, device=device)

    def __call__(self, X):
        for layer in self.layers:
            output = layer(X)

        return self.layer_out(output)


class MLP(torch.nn.Module):
    def __init__(
        self,
        n: int,
        num_output_list: List[int],
        dropout_list: List[int],
        block_io_shape: int = 64,
        outputs: int = 2,
        device: str = "cuda",
    ):
        super().__init__()

        self.layer_in = LazyLinear(block_io_shape, device=device)

        self.layers = [
            Block(
                num_output_list=num_output_list,
                dropout_list=dropout_list,
                num_output=block_io_shape,
            )
            for _ in range(n)
        ]

        self.output_layer = LazyLinear(outputs, device=device)

        self.apply(self._init)

        self.mse_loss = torch.nn.MSELoss()

    def _init(self, module):
        if type(module) is torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def __call__(self, X):
        output = self.layer_in(X)

        for l in self.layers:
            output = l(output) + output

        return self.output_layer(output)

    def loss(self, y, pred):
        return self.mse_loss(y, pred)
