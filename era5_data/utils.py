import pandas as pd
import numpy as np
import sys
import os
from typing import Tuple, Optional
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from era5_data.config import cfg
from era5_data import utils_data as utils_data

from torch.nn.modules.module import _addindent
import logging


def logger_info(logger_name, log_path="default_logger.log"):
    """set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    """
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print("LogHandlers exist!")
    else:
        print("LogHandlers setup!")
        level = logging.INFO
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d : %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


"""
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
"""


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


def visuailze(output, target, input, var, z, step, path):
    # levels = np.linspace(-30, 90, 9)
    variables = cfg.ERA5_UPPER_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    max_bias = _calc_max_bias(output[var, z, :, :], target[var, z, :, :])

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(input[var, z, :, :], cmap="coolwarm")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target[var, z, :, :], cmap="coolwarm")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(
        output[var, z, :, :], cmap="coolwarm"
    )  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        output[var, z, :, :] - target[var, z, :, :],
        cmap="coolwarm",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}_Z{}".format(step, variables[var], z)))


def visuailze_surface(output, target, input, var, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var = variables.index(var)
    fig = plt.figure(figsize=(16, 2))

    max_bias = _calc_max_bias(output[var, :, :], target[var, :, :])

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(input[var, :, :], cmap="coolwarm")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target[var, :, :], cmap="coolwarm")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt (Δ 24h)")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(
        output[var, :, :], cmap="coolwarm"
    )  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred (Δ 24h)")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        output[var, :, :] - target[var, :, :],
        cmap="coolwarm",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}".format(step, variables[var])))
    plt.close()


def visualize_windspeed(output, target, input, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var1 = variables.index("u10")
    var2 = variables.index("v10")

    wind_speed_input = torch.sqrt(input[var1, :, :] ** 2 + input[var2, :, :] ** 2)
    wind_speed_input = prepare_europe(wind_speed_input)

    wind_speed_output = torch.sqrt(output[var1, :, :] ** 2 + output[var2, :, :] ** 2)
    wind_speed_output = prepare_europe(wind_speed_output)

    wind_speed_target = torch.sqrt(target[var1, :, :] ** 2 + target[var2, :, :] ** 2)
    wind_speed_target = prepare_europe(wind_speed_target)

    max_bias = _calc_max_bias(wind_speed_output, wind_speed_target)

    fig = plt.figure(figsize=(12, 2))

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(wind_speed_input, cmap="coolwarm")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(wind_speed_target, cmap="coolwarm")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt (Δ 24h)")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(wind_speed_output, cmap="coolwarm")
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred (Δ 24h)")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(
        wind_speed_output - wind_speed_target,
        cmap="coolwarm",
        vmin=-max_bias,
        vmax=max_bias,
    )
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_{}".format(step, "wind_speed")))
    plt.close()


def visuailze_power(output, target, input, step, path):
    variables = cfg.ERA5_SURFACE_VARIABLES
    var1 = variables.index("u10")
    var2 = variables.index("v10")
    wind_speed = torch.sqrt(input[var1, :, :] ** 2 + input[var2, :, :] ** 2)

    wind_speed = prepare_europe(wind_speed)
    output = prepare_europe(output)
    target = prepare_europe(target)

    max_bias = _calc_max_bias(output, target)

    fig = plt.figure(figsize=(12, 2))

    ax_1 = fig.add_subplot(141)
    plot3 = ax_1.imshow(wind_speed, cmap="coolwarm")
    plt.colorbar(plot3, ax=ax_1, fraction=0.05, pad=0.05)
    ax_1.title.set_text("input[ws]")

    ax_2 = fig.add_subplot(142)
    plot2 = ax_2.imshow(target, cmap="coolwarm")
    plt.colorbar(plot2, ax=ax_2, fraction=0.05, pad=0.05)
    ax_2.title.set_text("gt[ws] (Δ 24h)")

    ax_3 = fig.add_subplot(143)
    plot1 = ax_3.imshow(output, cmap="coolwarm")  # , levels = levels, extend = 'min')
    plt.colorbar(plot1, ax=ax_3, fraction=0.05, pad=0.05)
    ax_3.title.set_text("pred[ws] (Δ 24h)")

    ax_4 = fig.add_subplot(144)
    plot4 = ax_4.imshow(output - target, cmap="coolwarm", vmin=-max_bias, vmax=max_bias)
    plt.colorbar(plot4, ax=ax_4, fraction=0.05, pad=0.05)
    ax_4.title.set_text("bias[ws]")

    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "{}_power".format(step)))
    plt.close()


def _plot_map(ax, data, title, color_coords, cmap="coolwarm", vmin=None, vmax=None):
    """Helper function to plot a single map with Cartopy."""
    # visible_coords = [-22, 38.5, 31, 69]  # [lon_min, lon_max, lat_min, lat_max]
    visible_coords = [-22, 45, 27, 72]

    ax.set_extent(visible_coords, crs=ccrs.PlateCarree())  # Crop to Europe
    ax.coastlines(resolution="50m", linewidth=0.3, color="#777777")
    ax.add_feature(
        cfeature.BORDERS, linewidth=0.3, edgecolor="#777777", facecolor="none"
    )

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-180, 181, 10),
        ylocs=np.arange(-90, 91, 10),
        linewidth=0.3,
        color="#777777",
        alpha=0.5,
    )
    gl.top_labels = False
    gl.right_labels = False

    # Flip data horizontally to match Cartopy's coordinate system
    data = torch.flip(data, dims=(0,))

    plot = ax.imshow(
        data,
        extent=color_coords,
        origin="lower",
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(plot, ax=ax, fraction=0.037, pad=0.05)
    ax.set_title(title, fontsize=10)


def visualize_all(
    output_power: torch.Tensor,
    target_power: torch.Tensor,
    input_pangu_surface: torch.Tensor,
    input_pangu_upper: torch.Tensor,
    output_pangu_surface: torch.Tensor,
    output_pangu_upper: torch.Tensor,
    target_pangu_surface: torch.Tensor,
    target_pangu_upper: torch.Tensor,
    step: str,
    path: str,
    input_power: Optional[torch.Tensor] = None,
    use_surface: bool = False,
    z: int = 0,
    epoch: Optional[int] = None,
):
    """Visualizes wind speeds (Pangu) and power predictions using Cartopy maps."""
    color_coords = [-22, 45.55, 26.4, 72.5]  # [lon_min, lon_max, lat_min, lat_max]

    # Extract wind speed data
    if use_surface:
        variables_surface = cfg.ERA5_SURFACE_VARIABLES
        var_u_surface = variables_surface.index("u10")
        var_v_surface = variables_surface.index("v10")
        input_ws = _calc_wind_speed(
            input_pangu_surface[var_u_surface, :, :],
            input_pangu_surface[var_v_surface, :, :],
        )
        target_ws = _calc_wind_speed(
            target_pangu_surface[var_u_surface, :, :],
            target_pangu_surface[var_v_surface, :, :],
        )
        output_ws = _calc_wind_speed(
            output_pangu_surface[var_u_surface, :, :],
            output_pangu_surface[var_v_surface, :, :],
        )
    else:
        variables_upper = cfg.ERA5_UPPER_VARIABLES
        var_u_upper = variables_upper.index("u")
        var_v_upper = variables_upper.index("v")
        input_ws = _calc_wind_speed(
            input_pangu_upper[var_u_upper, z, :, :],
            input_pangu_upper[var_v_upper, z, :, :],
        )
        target_ws = _calc_wind_speed(
            target_pangu_upper[var_u_upper, z, :, :],
            target_pangu_upper[var_v_upper, z, :, :],
        )
        output_ws = _calc_wind_speed(
            output_pangu_upper[var_u_upper, z, :, :],
            output_pangu_upper[var_v_upper, z, :, :],
        )

    # Prepare data for visualization
    input_ws = prepare_europe(input_ws)
    target_ws = prepare_europe(target_ws)
    output_ws = prepare_europe(output_ws)
    target_power = prepare_europe(target_power)
    output_power = prepare_europe(output_power)
    if input_power is not None:
        input_power = prepare_europe(input_power)

    max_bias_ws = _calc_max_bias(output_ws, target_ws)
    max_bias_power = _calc_max_bias(output_power, target_power)

    # Create figure with subplots
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 4),
        dpi=300,
        subplot_kw={
            "projection": ccrs.Orthographic(
                central_longitude=11.5, central_latitude=49.5
            )
        },
    )

    # Wind Speed Subplots
    _plot_map(axes[0, 0], input_ws, "input[wind speed]", color_coords)
    _plot_map(axes[0, 1], target_ws, "gt[wind speed] Δ24h", color_coords)
    _plot_map(axes[0, 2], output_ws, "pred[wind speed] Δ24h", color_coords)
    _plot_map(
        axes[0, 3],
        output_ws - target_ws,
        "bias[wind speed]",
        color_coords,
        vmin=-max_bias_ws,
        vmax=max_bias_ws,
    )

    # Power Subplots
    # Only plot input power if it is provided
    if input_power is not None:
        _plot_map(axes[1, 0], input_power, "input[power]", color_coords)
    else:
        fig.delaxes(axes[1, 0])  # Remove the axis if input_power is not provided
    _plot_map(axes[1, 1], target_power, "gt[power]", color_coords)
    _plot_map(axes[1, 2], output_power, "pred[power] Δ24h", color_coords)
    _plot_map(
        axes[1, 3],
        output_power - target_power,
        "bias[power]",
        color_coords,
        vmin=-max_bias_power,
        vmax=max_bias_power,
    )

    plt.tight_layout()
    if epoch is None:
        plt.savefig(os.path.join(path, f"{step}_3d_power.pdf"))
    else:
        plt.savefig(os.path.join(path, f"{step}_3d_power_epoch{epoch}.pdf"))
    plt.close()


def load_pangu_output(step: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load pangu outputs for a given step (pre-generated pangu outputs).

    Parameters
    ----------
    step : str
        Target step (date & time): e.g. "2017051000" (YYYYMMDDHH).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Upper and surface level outputs which were generated by the pangu baseline model (24h).
    """
    output_path = cfg.PANGU_INFERENCE_OUTPUTS
    output_upper = torch.load(
        os.path.join(output_path, f"output_upper_{step}.pth"), weights_only=False
    )
    output_surface = torch.load(
        os.path.join(output_path, f"output_surface_{step}.pth"), weights_only=False
    )
    return output_upper, output_surface


def _calc_max_bias(output, target):
    """Calculate the maximum bias between the output and target. Used for bias color scale"""
    bias = output - target
    bias_masked = bias[~torch.isnan(bias)]
    max_bias = torch.max(torch.abs(bias_masked)).item()
    return max_bias


def _calc_wind_speed(u, v):
    return torch.sqrt(u**2 + v**2)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def torch_summarize(
    model, show_weights=False, show_parameters=False, show_gradients=False
):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + " (\n"
    total_params = sum(
        [
            np.prod(p.size())
            for p in filter(lambda p: p.requires_grad, model.parameters())
        ]
    )
    tmpstr += ", total parameters={}".format(total_params)
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )
        weights = tuple(
            [
                tuple(p.size())
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )
        grads = tuple(
            [
                str(p.requires_grad)
                for p in filter(lambda p: p.requires_grad, module.parameters())
            ]
        )

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        if show_gradients:
            tmpstr += ", gradients={}".format(grads)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    return tmpstr


def save_errorScores(csv_path, z, q, t, u, v, surface, error):
    score_upper_z = pd.DataFrame.from_dict(
        z, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_q = pd.DataFrame.from_dict(
        q, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_t = pd.DataFrame.from_dict(
        t, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_u = pd.DataFrame.from_dict(
        u, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_upper_v = pd.DataFrame.from_dict(
        v, orient="index", columns=cfg.ERA5_UPPER_LEVELS
    )
    score_surface = pd.DataFrame.from_dict(
        surface, orient="index", columns=cfg.ERA5_SURFACE_VARIABLES
    )

    score_upper_z.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_z"))
    score_upper_q.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_q"))
    score_upper_t.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_t"))
    score_upper_u.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_u"))
    score_upper_v.to_csv("{}/{}.csv".format(csv_path, f"{error}_upper_v"))
    score_surface.to_csv("{}/{}.csv".format(csv_path, f"{error}_surface"))


def save_error_power(csv_path, power_scores, error):
    score_power = pd.DataFrame.from_dict(
        power_scores, orient="index", columns=["power"]
    )
    score_power.to_csv("{}/{}.csv".format(csv_path, f"{error}_power"))


def prepare_europe(data: torch.Tensor, fill_value=float("nan")) -> torch.Tensor:
    """Cut out Europe area from the data and replace land area with NaN."""
    lsm = utils_data.loadLandSeaMask(
        device=None, mask_type="sea", fill_value=fill_value
    )
    # Cut out Europe area
    data = data * lsm
    data = data.squeeze()
    data = torch.roll(data, shifts=88, dims=1)
    data = torch.roll(data, shifts=-70, dims=0)
    data = data[0:185, 0:271]
    return data


if __name__ == "__main__":
    """

    s_transforms = []

    s_transforms.append(T.RandomHorizontalFlip())

    s_transforms.append(T.RandomVerticalFlip())
    s_transforms = T.Compose(s_transforms)
    s_transforms = None

    nc_dataset = NetCDFDataset(dataset_path,
                               data_transform=None,
                               training=False,
                               validation = True,
                               startDate = '20150101',
                               endDate='20150102',
                               freq='H',
                               horizon=5)
    nc_dataloader = data.DataLoader(dataset=nc_dataset, batch_size=2,
                                          drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    print('Total length is', len(nc_dataset))


    start_time = time.time()
    nc_dataloader = iter(nc_dataloader)
    for i in range(2):
        input, input_surface, target, target_surface, periods = next(nc_dataloader)
        print(input.shape) #torch.Size([1, 5, 13, 721, 1440])
        print(input_surface.shape) #torch.Size([1, 4, 721, 1440])
        print(target.shape) #torch.Size([1, 5, 13, 721, 1440])
        print(target_surface.shape) #torch.Size([1, 4, 721, 1440])


        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        plot1 = ax1.contourf(input[0,0,0])
        plt.colorbar(plot1,ax=ax1)
        ax1.title.set_text('input 0')

        ax2 = fig.add_subplot(222)
        plot2 = ax2.contourf(input_surface[0,0].squeeze())
        plt.colorbar(plot2,ax=ax2)
        ax2.title.set_text('input_surface 0')

        ax3 = fig.add_subplot(223)
        plot3 = ax3.contourf(target[0,0,0].squeeze())
        ax3.title.set_text('target 0')
        plt.colorbar(plot3,ax=ax3)


        ax4 = fig.add_subplot(224)
        plot4 = ax4.contourf(target_surface[0,0].squeeze())
        ax4.title.set_text('target_surface 0')
        plt.colorbar(plot4,ax=ax4)
        plt.tight_layout()
        plt.savefig(fname='compare_{}_{}'.format(periods[0], periods[1]))
        print("image saved!")


    elapsed = time.time() - start_time
    elapsed = str(timedelta(seconds=elapsed))
    print("Elapsed [{}]".format(elapsed))
    """
