import os
from datetime import datetime
import warnings
from era5_data import utils
from wind_fusion.pangu_pytorch.models.train_power import (
    load_constants,
    model_inference,
    load_land_sea_mask,
    visualize,
)


warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
)

warnings.filterwarnings(
    "ignore",
    message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
)


def test(test_loader, model, device, res_path):
    aux_constants = load_constants(device)
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
        output_power_test, output_surface_test = model_inference(
            model, input_test, input_surface_test, aux_constants
        )
        lsm_expanded = load_land_sea_mask(output_power_test.device)
        output_power_test = output_power_test * lsm_expanded
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
