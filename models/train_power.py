import os
import copy
import torch
from torch import nn
import torch.distributed as dist
import warnings
from wind_fusion.pangu_pytorch.era5_data import utils, utils_data
from wind_fusion.pangu_pytorch.models.baseline_formula import BaselineFormula
from wind_fusion.pangu_pytorch.era5_data.config import cfg
from typing import Tuple, Dict, List, Union, Optional
import logging
from tensorboardX import SummaryWriter


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)

warnings.filterwarnings(
    "ignore",
    message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
)


def load_land_sea_mask(
    device: torch.device, mask_type: str = "sea", fill_value: int = 0
) -> torch.Tensor:
    """
    Load the land-sea mask. Used to mask out land points in the loss calculation.

    Parameters
    ----------
    device : torch.device
        The device to load the mask onto.
    mask_type : str, optional
        The type of mask to load, by default "sea", can also be "land".
    fill_value : int, optional
        The value to fill the rest of the mask with, by default 0.

    Returns
    -------
    torch.Tensor
        The loaded land-sea mask.
    """
    return utils_data.loadLandSeaMask(
        device, mask_type=mask_type, fill_value=fill_value
    )


def model_inference_power(
    model: nn.Module,
    input: torch.Tensor,
    input_surface: torch.Tensor,
    aux_constants: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Inference code for power models."""
    output_power = model(
        input,
        input_surface,
        aux_constants["weather_statistics"],
        aux_constants["constant_maps"],
        aux_constants["const_h"],
    )

    return output_power


def model_inference_pangu(
    model: nn.Module,
    input: torch.Tensor,
    input_surface: torch.Tensor,
    aux_constants: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference for the Pangu model. Pangu outputs are normalized."""
    output_upper, output_surface = model(
        input,
        input_surface,
        aux_constants["weather_statistics"],
        aux_constants["constant_maps"],
        aux_constants["const_h"],
    )

    # Transfer to the output to the original data range
    output_upper, output_surface = utils_data.normBackData(
        output_upper, output_surface, aux_constants["weather_statistics_last"]
    )

    return output_upper, output_surface


def baseline_inference(
    input_power: torch.Tensor,
    mean_power: torch.Tensor,
    output_upper: Optional[torch.Tensor] = None,
    output_surface: Optional[torch.Tensor] = None,
    baseline_formula: Optional[BaselineFormula] = None,
    type: str = "persistence",
) -> torch.Tensor:
    """Returns the specified baseline prediction.
    Persistence: returns the input power as the prediction.
    Mean: returns the mean power per grid point as the prediction.
    Formula: returns the prediction using the formula model.

    Parameters
    ----------
    input_power : torch.Tensor
        Power capacity factor at time t.
    mean_power : torch.Tensor
        A tensor containing the mean power per grid point.
    output_upper : Optional[torch.Tensor], optional
    The upper-level pangu model output used for the formula model, by default None.
    output_surface : Optional[torch.Tensor], optional
        The surface-level pangu model output used for the formula model, by default None.
    baseline_formula: Optional[BaselineFormula], optional
        The formula model used for the formula baseline, by default None.
    type : str, optional
        Specifies the type of baseline prediction, by default "persistence".

    Returns
    -------
    torch.Tensor
        Forecasted power capacity factor at time t+1.
    """

    if type == "persistence":
        return input_power
    elif type == "mean":
        return mean_power
    elif type == "formula":
        assert (
            output_surface is not None
        ), "output_surface must be provided for formula baseline."
        assert (
            output_upper is not None
        ), "output_upper must be provided for formula baseline."
        assert (
            baseline_formula is not None
        ), "baseline_formula model must be provided for formula baseline."

        # baseline_formula.eval()
        return baseline_formula(output_upper, output_surface)

    raise NotImplementedError(f"Baseline type {type} not implemented.")


def calculate_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    lsm_expanded: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the loss for the model output. Applies the land-sea mask to the output and target before calculating the loss.

    Parameters
    ----------
    output : torch.Tensor
        The model output.
    target : torch.Tensor
        The target values.
    criterion : nn.Module
        The loss criterion.
    lsm_expanded : torch.Tensor
        The land-sea mask.

    Returns
    -------
    torch.Tensor
        The calculated loss.
    """
    mask_not_zero = ~(lsm_expanded == 0)
    mask_not_zero = mask_not_zero.unsqueeze(1)
    output = output * lsm_expanded
    loss = criterion(output[mask_not_zero], target[mask_not_zero])
    return torch.mean(loss)


def visualize(
    output_power: torch.Tensor,
    target_power: torch.Tensor,
    input_surface: torch.Tensor,
    input_upper: torch.Tensor,
    target_surface: torch.Tensor,
    target_upper: torch.Tensor,
    step: str,
    path: str,
    input_power: Optional[torch.Tensor] = None,
    epoch: Optional[int] = None,
) -> None:
    """For documentation, see utils.visuailze_all function"""
    if input_power is not None:
        input_power = input_power.detach().cpu().squeeze()

    # Load pre-generated pangu outputs for visualization
    output_upper, output_surface = utils.load_pangu_output(step)

    utils.visuailze_all(
        output_power.detach().cpu().squeeze(),
        target_power.detach().cpu().squeeze(),
        input_surface.detach().cpu().squeeze(),
        input_upper.detach().cpu().squeeze(),
        output_surface.detach().cpu().squeeze(),
        output_upper.detach().cpu().squeeze(),
        target_surface.detach().cpu().squeeze(),
        target_upper.detach().cpu().squeeze(),
        step=step,
        path=path,
        input_power=input_power,
        epoch=epoch,
    )


def save_output_pth(
    output_upper: torch.Tensor,
    output_surface: torch.Tensor,
    target_time: str,
    res_path: str,
) -> None:
    """
    Save the pangu output tensors to .pth files, those are used visualization purposes.
    Only required, when the pangu outputs are not pre-generated (e.g., when test date range changes).
    Parameters
    ----------
    output_upper : torch.Tensor
        The tensor containing the upper level pangu output data.
    output_surface : torch.Tensor
        The tensor containing the surface level pangu output data.
    target_time : str
        The target time string used to name the output files.
    res_path : str
        The path to the directory where the output files will be saved.
    Returns
    -------
    None
    """

    output_path = os.path.join(res_path, "model_output")
    utils.mkdirs(output_path)
    torch.save(
        output_upper, os.path.join(output_path, f"output_upper_{target_time}.pth")
    )
    torch.save(
        output_surface, os.path.join(output_path, f"output_surface_{target_time}.pth")
    )


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    res_path: str,
    writer: SummaryWriter,
    logger: logging.Logger,
    start_epoch: int,
    rank: int = 0,
    device: Union[torch.device, None] = None,
) -> nn.Module:
    """
    Train the model using the provided training and validation data loaders, optimizer, and learning rate scheduler.

    This function performs training over a specified number of epochs, validates the model at specified intervals,
    saves model checkpoints, and implements early stopping if there is no improvement in validation loss for a
    certain number of epochs. The training process is synchronized across multiple ranks for distributed training.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    lr_scheduler : torch.optim.lr_scheduler.MultiStepLR
        Learning rate scheduler.
    res_path : str
        Path to save model checkpoints.
    writer : SummaryWriter
        TensorBoard SummaryWriter for logging.
    logger : logging.Logger
        Logger for logging training progress and information.
    start_epoch : int
        The epoch to start training from.
    rank : int, optional
        Rank of the current process in distributed training, by default 0.
    device : Union[torch.device, None], optional
        Device to run the training on, by default None.

    Returns
    -------
    nn.Module
        The best model based on validation loss.
    """
    criterion = nn.L1Loss(reduction="none")
    epochs = cfg.PG.TRAIN.EPOCHS
    loss_list: List[float] = []
    best_loss = float("inf")
    epochs_since_last_improvement = 0
    best_model = model
    aux_constants = utils_data.loadAllConstants(device=device)

    # Termination flag to signal early stopping
    early_stop_flag = torch.tensor(
        [0], dtype=torch.int, device=device
    )  # 0 means continue, 1 means stop

    for i in range(start_epoch, epochs + 1):
        if early_stop_flag.item() == 1:
            break  # If early stop flag is set, break out of the loop

        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            aux_constants,
            logger,
            rank,
            device,
            i,
        )
        loss_list.append(epoch_loss)
        lr_scheduler.step()

        if rank == 0 and i % cfg.PG.TRAIN.SAVE_INTERVAL == 0:
            save_model_checkpoint(model, optimizer, lr_scheduler, res_path, i)

        # Save the last model checkpoint
        if rank == 0:
            save_model_checkpoint(
                model, optimizer, lr_scheduler, res_path, i, type="last"
            )

        # Validate on all ranks (on purpose, since barrier times out if only rank 0 validates)
        # TODO(EliasKng): Use DistributedSampler to spread validation across all ranks, then calculate mean loss
        if i % cfg.PG.VAL.INTERVAL == 0:
            val_loss, best_model, epochs_since_last_improvement = validate(
                model,
                optimizer,
                lr_scheduler,
                val_loader,
                criterion,
                aux_constants,
                writer,
                logger,
                res_path,
                best_loss,
                epoch_loss,
                best_model,
                epochs_since_last_improvement,
                rank,
                device,
                i,
            )

            # Set best loss
            if val_loss < best_loss:
                best_loss = val_loss

            # Set early stop flag
            if rank == 0 and epochs_since_last_improvement >= 5:
                logger.info(
                    f"No improvement in validation loss for {epochs_since_last_improvement} epochs, terminating training."
                )
                early_stop_flag[0] = 1  # Set the early stop flag

        # Broadcast early stop flag from rank 0 to all other ranks
        dist.broadcast(early_stop_flag, src=0)

        # Synchronize all ranks
        dist.barrier()

    return best_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    aux_constants: Dict[str, torch.Tensor],
    logger: logging.Logger,
    rank: int,
    device: Union[torch.device, None],
    epoch: int,
) -> float:
    """
    Trains the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    optimizer : torch.optim.Optimizer
        The optimizer for training.
    criterion : nn.Module
        The loss criterion.
    aux_constants : Dict[str, torch.Tensor]
        Auxiliary constants for the model. Contains weather statistics and constant maps.
    logger : logging.Logger
        The logger for logging information.
    rank : int
        The rank of the current process.
    device : Union[torch.device, None]
        The device to train on.
    epoch : int
        The current epoch number.

    Returns
    -------
    float
        The average loss for the epoch.
    """
    epoch_loss = 0.0
    print(f"Starting epoch {epoch}/{cfg.PG.TRAIN.EPOCHS}")

    for id, train_data in enumerate(train_loader):
        (
            input,
            input_surface,
            input_power,
            target_power,
            target_upper,
            target_surface,
            periods,
        ) = train_data
        input, input_surface, target_power = (
            input.to(device),
            input_surface.to(device),
            target_power.to(device),
        )
        print(f"(T) Processing batch {id + 1}/{len(train_loader)}")

        optimizer.zero_grad()
        model.train()

        # Model inference
        output_power = model_inference_power(model, input, input_surface, aux_constants)

        # Load lsm and calculate loss
        lsm_expanded = load_land_sea_mask(output_power.device)
        loss = calculate_loss(output_power, target_power, criterion, lsm_expanded)

        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch} finished with training loss: {epoch_loss:.4f}")
    if rank == 0:
        logger.info("Epoch {} : {:.3f}".format(epoch, epoch_loss))

    return epoch_loss


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    res_path: str,
    epoch: int,
    type: str = "train",
) -> None:
    """
    Save the model checkpoint to a specified directory.

    Parameters
    ----------
    model : nn.Module
        The model to be saved.
    optimizer : torch.optim.Optimizer
        The optimizer associated with the model.
    lr_scheduler : torch.optim.lr_scheduler.MultiStepLR
        The learning rate scheduler associated with the model.
    res_path : str
        The directory path where the model checkpoint will be saved.
    epoch : int
        The current epoch number.
    type : str, optional
        The type of checkpoint to save (only adapts the checkpoint name). Options are "train", "best", or "last".
        Default is "train".

    Returns
    -------
    None
    """
    model_save_path = os.path.join(res_path, "models")
    utils.mkdirs(model_save_path)
    save_file = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    # Use different file name for best model so the last one will be overwritten
    if type == "best":
        file_name = "best_checkpoint.pth"
    elif type == "train":
        file_name = "train_{}.pth".format(epoch)
    elif type == "last":
        file_name = "last.pth"
    torch.save(save_file, os.path.join(model_save_path, file_name))
    print(f"Model saved at epoch {epoch}: {file_name}")


def validate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    aux_constants: Dict[str, torch.Tensor],
    writer: SummaryWriter,
    logger: logging.Logger,
    res_path: str,
    best_loss: float,
    epoch_loss: float,
    best_model: nn.Module,
    epochs_since_last_improvement: int,
    rank: int,
    device: Union[torch.device, None],
    epoch: int,
) -> Tuple[float, nn.Module, int]:
    """
    Validate the model on the validation dataset and update the best model if the validation loss improves.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be validated.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.
    lr_scheduler : torch.optim.lr_scheduler.MultiStepLR
        The learning rate scheduler used during training.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        The loss function used to calculate the validation loss.
    aux_constants : Dict[str, torch.Tensor]
        Auxiliary constants required for model inference.
    writer : SummaryWriter
        TensorBoard SummaryWriter for logging.
    logger : logging.Logger
        Logger for logging validation information.
    res_path : str
        Path to save the results and model checkpoints.
    best_loss : float
        The best validation loss achieved so far.
    epoch_loss : float
        The training loss for the current epoch.
    best_model : nn.Module
        The best model based on validation loss.
    epochs_since_last_improvement : int
        Number of epochs since the last improvement in validation loss. Needed for early stopping.
    rank : int
        Rank of the current process (used for distributed training).
    device : Union[torch.device, None]
        Device to run the validation on.
    epoch : int
        The current epoch number.

    Returns
    -------
    Tuple[float, nn.Module, int]
        A tuple containing the validation loss, the best model, and the number of epochs since the last improvement.
    """

    print(f"Starting validation at epoch {epoch}")
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for id, val_data in enumerate(val_loader, 0):
            (
                input_upper_val,
                input_surface_val,
                input_power_val,
                target_power_val,
                target_upper_val,
                target_surface_val,
                periods_val,
            ) = val_data
            input_upper_val, input_surface_val, target_power_val = (
                input_upper_val.to(device),
                input_surface_val.to(device),
                target_power_val.to(device),
            )
            print(f"(V) Processing batch {id + 1}/{len(val_loader)}")
            output_power_val = model_inference_power(
                model, input_upper_val, input_surface_val, aux_constants
            )
            lsm_expanded = load_land_sea_mask(output_power_val.device)
            loss = calculate_loss(
                output_power_val, target_power_val, criterion, lsm_expanded
            )
            val_loss += loss.item()

        val_loss /= len(val_loader)
        if rank == 0:
            writer.add_scalars("Loss", {"train": epoch_loss, "val": val_loss}, epoch)
            logger.info("Validate at Epoch {} : {:.3f}".format(epoch, val_loss))
            png_path = os.path.join(res_path, "png_training")
            utils.mkdirs(png_path)
            visualize(
                output_power_val,
                target_power_val,
                input_surface_val,
                input_upper_val,
                target_surface_val,
                target_upper_val,
                periods_val[1][0],
                png_path,
                epoch=epoch,
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.module)
            # Save both a deepcopy and statedict of the best model (deepcopy is for testing, statedict for re-using model)
            if rank == 0:
                save_model_checkpoint(
                    model, optimizer, lr_scheduler, res_path, epoch, type="best"
                )
                torch.save(
                    best_model, os.path.join(res_path, "models", "best_model.pth")
                )
                logger.info(
                    f"New best model saved at epoch {epoch} with validation loss: {val_loss:.4f}"
                )
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1

    return val_loss, best_model, epochs_since_last_improvement
