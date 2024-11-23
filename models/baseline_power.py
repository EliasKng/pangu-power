from typing import Tuple
from torch import Tensor
from torch import nn


class BaselineModel_Persistence(nn.Module):
    """A baseline persistence model. It simply returns the input as the output."""

    # @TODO(EliasKng): Requires input to be energy data instead of weather data. Fix.
    def forward(
        self,
        input: Tensor,
        input_surface: Tensor,
        # Inputs not used but kept for compatibility
        statistics: Tensor,
        maps: Tensor,
        const_h: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output = input
        output_surface = input_surface

        return output, output_surface
