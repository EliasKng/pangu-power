import sys
import os
from typing import Tuple
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch import nn


class BaselineFormula(nn.Module):
    """Baseline model that uses wind turbine power curve to predict power"""

    def __init__(self):
        # TODO(EliasKng): Move to config
        self.offshore_power_curve_fapacity_factor = {
            0: 0.0,
            3.5: 0.0,
            4: 0.00875,
            4.5: 0.01875,
            5: 0.035,
            5.5: 0.063125,
            6: 0.09375,
            6.5: 0.1375,
            7: 0.18125,
            7.5: 0.240625,
            8: 0.3,
            8.5: 0.38625,
            9: 0.4725,
            9.5: 0.58625,
            10: 0.7,
            10.5: 0.79875,
            11: 0.8975,
            11.5: 0.95,
            12: 0.96875,
            12.5: 0.990625,
            13: 1.0,
            25: 1.0,
            25.000000001: 0.0,
            500: 0.0,
        }

        # Check if keys are sorted, required for linear search in interpolation
        assert list(self.offshore_power_curve_fapacity_factor.keys()) == sorted(
            self.offshore_power_curve_fapacity_factor.keys()
        ), "Keys of offshore_power_curve_fapacity_factor are not sorted"

        # Prepare lists of wind speed and power
        self.wind_speeds = torch.tensor(
            list(self.offshore_power_curve_fapacity_factor.keys()), dtype=torch.float32
        )
        self.power_levels = torch.tensor(
            list(self.offshore_power_curve_fapacity_factor.values()),
            dtype=torch.float32,
        )

    def forward(
        self,
        pangu_output_surface: Tensor,
        pangu_output_upper: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # TODO(EliasKng): Isolate wind speed components (u and v), then calculate wind speed, then calculate power

        output_power = None
        return output_power

    # Interpolation function
    def _interpolate_wind_capacity_factor(
        self, wind_speed: torch.Tensor
    ) -> torch.Tensor:
        """Interpolates the wind power capacity factor tensor for a given wind speed tensor.
        The interpolation is done using the power curve data of the wind turbine (Power [kW] - Vestas Offshore V164-8000).

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing wind speeds.

        Returns
        -------
        torch.Tensor
            A tensor containing interpolated wind power values.
        """
        # Clamp the input within the range of wind_speeds
        wind_speed = torch.clamp(
            wind_speed, min=self.wind_speeds.min(), max=self.wind_speeds.max()
        )
        # Find indices of the interval
        indices = torch.searchsorted(self.wind_speeds, wind_speed, right=True)
        indices = torch.clamp(indices, 1, len(self.wind_speeds) - 1)

        x0 = self.wind_speeds[indices - 1]
        x1 = self.wind_speeds[indices]
        y0 = self.powers[indices - 1]
        y1 = self.powers[indices]

        # Linear interpolation
        return y0 + (y1 - y0) * (wind_speed - x0) / (x1 - x0)
