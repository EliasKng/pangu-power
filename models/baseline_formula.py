import sys
import os
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch import nn
from wind_fusion.pangu_pytorch.era5_data.config import cfg


class BaselineFormula(nn.Module):
    """Baseline model that uses wind turbine power curve to predict power"""

    def __init__(self, device: torch.device):
        super().__init__()
        self.offshore_power_curve_fapacity_factor = cfg.POWER_CURVE_OFFSHORE

        # Check if keys are sorted, required for linear search in interpolation
        assert list(self.offshore_power_curve_fapacity_factor.keys()) == sorted(
            self.offshore_power_curve_fapacity_factor.keys()
        ), "Keys of offshore_power_curve_fapacity_factor are not sorted"

        # Prepare lists of wind speed and power
        self.wind_speeds = torch.tensor(
            list(self.offshore_power_curve_fapacity_factor.keys()),
            dtype=torch.float32,
            device=device,
        )
        self.power_levels = torch.tensor(
            list(self.offshore_power_curve_fapacity_factor.values()),
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        pangu_output_upper: Tensor,
        pangu_output_surface: Tensor,
    ) -> Tensor:
        # Calculate wind speed from u and v components (surface level). ws = (u^2 + v^2)^0.5
        wind_speed = torch.sqrt(
            pangu_output_surface[:, 1, :, :] ** 2
            + pangu_output_surface[:, 2, :, :] ** 2
        )

        # Calculate wind power
        output_power = self._interpolate_wind_capacity_factor(wind_speed)

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
        y0 = self.power_levels[indices - 1]
        y1 = self.power_levels[indices]

        # Linear interpolation
        return y0 + (y1 - y0) * (wind_speed - x0) / (x1 - x0)
