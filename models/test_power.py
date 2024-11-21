import os
from datetime import datetime
import warnings
from era5_data import utils
from wind_fusion.pangu_pytorch.models.train_power import (
    model_inference,
    load_land_sea_mask,
    visualize,
)
from era5_data import utils_data, score


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)

warnings.filterwarnings(
    "ignore",
    message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
)


def test(test_loader, model, device, res_path):
    rmse_power = dict()
    # acc_power = dict()

    aux_constants = utils_data.loadAllConstants(device=device)

    for id, data in enumerate(test_loader, 0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] predict on {id}")
        (
            input_test,
            input_surface_test,
            target_power_test,
            target_upper_test,
            target_surface_test,
            periods_test,
        ) = data

        input_test, input_surface_test, target_power_test = (
            input_test.to(device),
            input_surface_test.to(device),
            target_power_test.to(device),
        )
        model.eval()

        # Inference
        output_power_test, output_surface_test = model_inference(
            model, input_test, input_surface_test, aux_constants
        )

        # Apply lsm
        lsm_expanded = load_land_sea_mask(output_power_test.device)
        print("Make sure LSM contains NAs and not 0s")
        # TODO(EliasKng): Check if this is correct
        output_power_test = output_power_test * lsm_expanded

        # Visualize
        target_time = periods_test[1][0]
        png_path = os.path.join(res_path, "png")
        utils.mkdirs(png_path)
        visualize(
            output_power_test,
            target_power_test,
            input_surface_test,
            output_surface_test,
            target_surface_test,
            target_time,
            png_path,
        )

        # Compute test scores
        # RMSE
        output_power_test = output_power_test.squeeze()
        target_power_test = target_power_test.squeeze()

        rmse_power[target_time] = score.rmse(output_power_test, target_power_test)
        # TODO(EliasKng): Implement score calculation
