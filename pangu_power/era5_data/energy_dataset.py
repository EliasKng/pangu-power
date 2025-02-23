from typing import List, Tuple
import xarray as xr
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class EnergyDataset(Dataset):
    def __init__(
        self,
        filepath_era5: str,
        filepath_power: str,
        startDate="20150101",
        endDate="20150102",
        freq="h",
        horizon=24,
        seed=1234,
    ) -> None:
        """
        Parameters
        ----------
        filepath_era5 : str
            Filepath to the ERA5 dataset (zarr).
        filephath_power : str
            Filepath to the power dataset (zarr).
        """
        # Load ERA5 and power datasets
        self.era5_upper, self.era5_surface = self._get_era5_data(filepath_era5)
        self.power = self._load_power_data(filepath_power)

        # Generate list of datetime keys based on the specified range and frequency
        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))

        # Calculate the length of the dataset
        self.length = len(self.keys) - horizon // 12 - 1
        self.horizon = horizon

        # Set the random seed for reproducibility
        random.seed(seed)

    def _load_data(
        self, key: datetime
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Tuple[str, str],
    ]:
        """
        Load data for a given datetime key.

        This method retrieves the input ERA5 datasets (upper and surface), the target power dataset,
        and the target ERA5 datasets (upper and surface) for the specified datetime key. It also calculates
        the target time by adding the horizon to the start time. The power dataset is reindexed to have the
        same longitudes as the input datasets. The datasets are then converted to numpy arrays.

        Parameters
        ----------
        key : datetime
            The datetime key for which to load the data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]
            A tuple containing the input upper dataset, input surface dataset, input&target power dataset,
            target upper ERA5 dataset, target surface ERA5 dataset, and a tuple of start and end time strings.
        """
        # start_time datetime obj
        start_time = key
        # convert datetime obj to string for matching file name and return key
        start_time_str = datetime.strftime(key, "%Y%m%d%H")

        # target time = start time + horizon
        end_time = key + timedelta(hours=self.horizon)
        end_time_str = end_time.strftime("%Y%m%d%H")

        # Get era5 datasets (input)
        input_surface_dataset = self.era5_surface.sel(time=start_time)
        input_upper_dataset = self.era5_upper.sel(time=start_time)
        target_surface_dataset = self.era5_surface.sel(time=end_time)
        target_upper_dataset = self.era5_upper.sel(time=end_time)

        # Get power datasets (target) and force the target dataset to have the same longitudes as the input dataset
        input_dataset_power = self.power.sel(time=start_time)
        input_dataset_power = input_dataset_power.reindex(
            longitude=input_surface_dataset["longitude"].values,
            latitude=input_surface_dataset["latitude"].values,
            method=None,
        )
        input_dataset_power = input_dataset_power.fillna(0)

        target_dataset_power = self.power.sel(time=end_time)
        target_dataset_power = target_dataset_power.reindex(
            longitude=input_surface_dataset["longitude"].values,
            latitude=input_surface_dataset["latitude"].values,
            method=None,
        )
        target_dataset_power = target_dataset_power.fillna(0)

        # datasets to numpy
        input, input_surface = self._xr_era5_to_numpy(
            input_upper_dataset, input_surface_dataset
        )
        target_upper, target_surface = self._xr_era5_to_numpy(
            target_upper_dataset, target_surface_dataset
        )

        input_power = self._xr_power_to_numpy(input_dataset_power)
        target_power = self._xr_power_to_numpy(target_dataset_power)

        return (
            input,
            input_surface,
            input_power,
            target_power,
            target_upper,
            target_surface,
            (start_time_str, end_time_str),
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Tuple[str, str],
    ]:
        """Returns input frames, target frames, and its corresponding time steps."""
        iii = self.keys[index]
        (
            input,
            input_surface,
            input_power,
            target_power,
            target_upper_pangu,
            target_surface_pangu,
            periods,
        ) = self._load_data(iii)

        return (
            input,
            input_surface,
            input_power,
            target_power,
            target_upper_pangu,
            target_surface_pangu,
            periods,
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

    @staticmethod
    def _xr_era5_to_numpy(
        dataset_upper: xr.Dataset, dataset_surface: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input
            xr.Dataset upper, surface
        Return
            numpy array upper, surface
        """

        upper_z = dataset_upper["z"].values.astype(np.float32)  # (13,721,1440)
        upper_q = dataset_upper["q"].values.astype(np.float32)
        upper_t = dataset_upper["t"].values.astype(np.float32)
        upper_u = dataset_upper["u"].values.astype(np.float32)
        upper_v = dataset_upper["v"].values.astype(np.float32)
        upper = np.concatenate(
            (
                upper_z[np.newaxis, ...],
                upper_q[np.newaxis, ...],
                upper_t[np.newaxis, ...],
                upper_u[np.newaxis, ...],
                upper_v[np.newaxis, ...],
            ),
            axis=0,
        )
        assert upper.shape == (5, 13, 721, 1440)
        # levels in descending order, require new memory space
        upper = upper[:, ::-1, :, :].copy()

        surface_mslp = dataset_surface["msl"].values.astype(np.float32)  # (721,1440)
        surface_u10 = dataset_surface["u10"].values.astype(np.float32)
        surface_v10 = dataset_surface["v10"].values.astype(np.float32)
        surface_t2m = dataset_surface["t2m"].values.astype(np.float32)
        surface = np.concatenate(
            (
                surface_mslp[np.newaxis, ...],
                surface_u10[np.newaxis, ...],
                surface_v10[np.newaxis, ...],
                surface_t2m[np.newaxis, ...],
            ),
            axis=0,
        )
        assert surface.shape == (4, 721, 1440)

        return upper, surface

    @staticmethod
    def _xr_power_to_numpy(dataset):
        """
        Input
            xr.Dataset power
        Return
            numpy array power
        """

        power = dataset["wofcfr"].values.astype(np.float32)
        power = power[np.newaxis, ...]
        assert power.shape == (1, 721, 1440)

        return power

    @staticmethod
    def _merge_datasets(filepaths: List[str]) -> xr.Dataset:
        ds = xr.Dataset()

        for filepath in filepaths:
            ds = xr.merge(
                [ds, xr.open_dataset(filepath)], compat="equals", join="exact"
            )
        return ds

    @staticmethod
    def _get_era5_data(filepath_era5: str) -> Tuple[xr.Dataset, xr.Dataset]:
        era5_data = xr.open_dataset(filepath_era5, engine="zarr")
        variable_mapping = {
            "geopotential": "z",
            "specific_humidity": "q",
            "temperature": "t",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "mean_sea_level_pressure": "msl",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "2m_temperature": "t2m",
        }

        surface_variables = ["msl", "u10", "v10", "t2m"]
        upper_variables = ["z", "q", "t", "u", "v"]

        # Select and rename variables according to the mapping
        era5_data = era5_data[list(variable_mapping.keys())].rename(variable_mapping)

        input_upper_dataset = era5_data[upper_variables]
        input_surface_dataset = era5_data[surface_variables]

        return input_upper_dataset, input_surface_dataset

    @staticmethod
    def _load_power_data(filepath: str) -> xr.Dataset:
        """
        Load power data from a Zarr dataset file.

        This function reads a dataset from the specified file path using the Zarr engine,
        adjusts the longitude values to be within the range [0, 360), and sorts the dataset
        by longitude, since the range of longitudes is -22 to 45.5 degrees, which makes it impossible to merge
        with era5 data.

        Args:
            filepath (str): The path to the Zarr dataset file.

        Returns:
            xr.Dataset: The loaded and processed power dataset.
        """
        power = xr.open_dataset(filepath, engine="zarr")
        power["longitude"] = power["longitude"] % 360
        power = power.sortby("longitude")
        return power
