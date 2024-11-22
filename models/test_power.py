import os
from datetime import datetime
import warnings
from era5_data import utils, utils_data
from wind_fusion.pangu_pytorch.models.train_power import (
    model_inference,
    load_land_sea_mask,
    visualize,
)
from era5_data import score


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
    mape_power = dict()
    acc_power = dict()

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
        lsm_expanded = load_land_sea_mask(output_power_test.device, fill_value=0)
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
        output_power_test = output_power_test.squeeze()
        target_power_test = target_power_test.squeeze()

        # Mask
        output_power_test_masked = output_power_test[lsm_expanded.squeeze() == 1]
        target_power_test_masked = target_power_test[lsm_expanded.squeeze() == 1]

        # RMSE
        rmse_power[target_time] = (
            (score.rmse(output_power_test_masked, target_power_test_masked))
            .detach()
            .cpu()
            .numpy()
        )

        # Mean absolute percentage error (MAPE)
        mape_power[target_time] = (
            (score.mape(output_power_test_masked, target_power_test_masked))
            .detach()
            .cpu()
            .numpy()
        )

        # ACC
        mean_power_per_grid_point = utils_data.loadMeanPower(output_power_test.device)

        # Calculate power anomalies
        output_power_anomaly = output_power_test - mean_power_per_grid_point
        target_power_anomaly = target_power_test - mean_power_per_grid_point

        # Mask anomalies
        output_power_anomaly_masked = output_power_anomaly.squeeze(0)[
            lsm_expanded.squeeze() == 1
        ]
        target_power_anomaly_masked = target_power_anomaly.squeeze(0)[
            lsm_expanded.squeeze() == 1
        ]

        # Calculate ACC
        acc_power[target_time] = (
            (
                score.weighted_acc(
                    output_power_anomaly_masked.detach().cpu(),
                    target_power_anomaly_masked.detach().cpu(),
                    weighted=False,
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

    # Save scores to csv
    csv_path = os.path.join(res_path, "csv")
    utils.mkdirs(csv_path)
    utils.save_error_power(
        csv_path,
        rmse_power,
        "rmse",
    )
    utils.save_error_power(
        csv_path,
        mape_power,
        "mape",
    )
    utils.save_error_power(
        csv_path,
        acc_power,
        "acc",
    )
