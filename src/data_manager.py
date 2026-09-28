# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#%%

from logging import getLogger

import os
import torch
import torchvision.transforms as transforms
import torchvision

from src.data.load_variables import load_brasil_surf_var, load_era5_static_variables

import numpy as np
import pandas as pd

_GLOBAL_SEED = 0
logger = getLogger()

def init_data(
    transform,
    batch_size,
    surf_vars=['Tmax', 'Tmin', 'pr'],
    static_vars=['slt', 'geo'],
    lat_lim=None, lon_lim=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    dataset_samples=None,
    adj_prep_balance=True,
    split_val=False,
    val_years=(2023, 2024, 2025),
    data_source='era5',
    deseason_standardize=False
):

    if data_source == 'era5':
        dataset = Era5Dataset(transform=transform, surf_vars=surf_vars,
                              lat_lim=lat_lim, lon_lim=lon_lim,
                              split_val=split_val, val_years=val_years,
                              return_time_period=True,
                              deseason_standardize=deseason_standardize)

    else:
        dataset = BrazilWeatherDataset( transform=transform,
                                        surf_vars=surf_vars,
                                        static_vars=static_vars,
                                        lat_lim=lat_lim, lon_lim=lon_lim,
                                        n_samples=dataset_samples,
                                        adj_prep_balance=adj_prep_balance,
                                        split_val=split_val,
                                        val_years=val_years)
        
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers)
    
    logger.info('data loader created')

    return (data_loader, dist_sampler)


def make_transforms(
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.5, 1.0),
    focal_crop_scale=(0.3, 0.7),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
    norm_means=None,
    norm_stds=None
):

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.Normalize(
            norm_means,
            norm_stds)
    ])

    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.Normalize(
            norm_means, 
            norm_stds)
    ])


    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    
    return transform


class MultiViewTransform(object):

    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views



class BrazilWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, transform, surf_vars, static_vars=None, return_patches=False,
                 patch_size=40, patch_stride=20, lat_lim=None, lon_lim=None,n_samples=None,adj_prep_balance=True,split_val=False,
                 val_years=(1980, 2000),
                 return_time_period=False):

        self.transform = transform

        self.imgs, self.validation_imgs = self.load_images(surf_vars, static_vars, return_patches=return_patches,
                                     patch_size=patch_size, patch_stride=patch_stride,
                                     lat_lim=lat_lim, lon_lim=lon_lim,n_samples=n_samples,adj_prep_balance=adj_prep_balance,
                                     split_val=split_val, val_years=tuple(val_years))

        self.return_time_period = return_time_period

    def load_images(self, surf_vars, static_vars,
                    return_patches,patch_size, patch_stride,
                    lat_lim, lon_lim,n_samples,
                    adj_prep_balance,split_val,val_years):

        surf_vars_values, time, mask = load_brasil_surf_var(surf_vars,lat_lim=lat_lim,lon_lim=lon_lim,n_samples=n_samples) # ['Tmax','Tmin','pr']
    
        x_surf = torch.stack(tuple(surf_vars_values.values()), dim=1)
        
        if static_vars is not None:
            
            static_vars_values, lat, lon  = load_era5_static_variables(static_vars,mask=mask,lat_lim=lat_lim,lon_lim= lon_lim) #['slt','geo']

            x_static = torch.stack(tuple(static_vars_values.values()), dim=0)

            B, _, H, W = x_surf.shape

            x_static = x_static.expand((B, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=1)
        
        if return_patches: 
            _, V, _, _ = x_surf.shape

            patches = x_surf.unfold(2, patch_size,patch_stride).unfold(3, patch_size, patch_stride)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, V, patch_size, patch_size)
            patches = patches[~torch.isnan(patches).any(dim=(1, 2, 3))] #exclude nans (ocean)

            return patches.float()
        
        else:
            
            if split_val:

                times_index = pd.to_datetime(time)
                val_mask = times_index.year.isin(list(val_years))
                val_indices = np.where(val_mask)[0]
                train_indices = np.where(~val_mask)[0]

                validation = x_surf[val_indices].float()
                x_surf = x_surf[train_indices]

                self.time = times_index[train_indices]
                self.val_time = times_index[val_indices]
                
            else:
                validation = None
                self.time = pd.to_datetime(time)
                
                
            if (surf_vars[0] == 'pr') and adj_prep_balance:
                print("Adjusting precipitation balance...")
                x_surf, time = self.adjust_prep_balance(x_surf, time, total_percnt=0.05, val_crop=5)
                self.time = time
                
            return x_surf.float(), validation


    def adjust_prep_balance(self, data, time, total_percnt = 0.05,val_crop = 5):
        
        low_prep_samples = np.where(data.mean(dim=(1,2,3)) < val_crop)[0]
        other_samples = np.where(data.mean(dim=(1,2,3)) >= val_crop)[0]

        n_samples = int((total_percnt*data.shape[0]) // 1)
        samples = np.linspace(0, low_prep_samples.shape[0] - 1, n_samples, dtype=int)

        new_dataset = torch.cat((data[other_samples],data[low_prep_samples[samples]]))
        time_others, time_lowprep = np.array(time)[other_samples] , np.array(time)[low_prep_samples[samples]]
        new_time = np.concatenate((time_others, time_lowprep))

        sorted_time = pd.Series(new_time).sort_values()
        new_dataset = new_dataset[sorted_time.index]
        new_time = pd.to_datetime(sorted_time.values)
        
        return new_dataset, new_time

    def __getitem__(self, index):

        cimg = self.imgs[index]
        timg = self.transform(cimg)

        if self.return_time_period:
            return timg, torch.tensor(self.time[index].to_period('M').ordinal)
        else:
            return timg, torch.tensor(-1)

    def __len__(self):
        return self.imgs.shape[0]


class Era5Dataset(torch.utils.data.Dataset):
    """ERA5 dataset that loads hourly surface variables and aggregates them to daily.
    """

    # Per-variable hourly -> daily aggregator. mx2t = daily max, mn2t = daily min,
    # tp = daily accumulated total.
    DAILY_AGG = {'mx2t': 'max', 'mn2t': 'min', 'tp': 'sum'}

    # Each variable was downloaded into its own set of yearly files, so a single
    # '*.nc' glob would concatenate temperature and precipitation files into one
    # time axis with each variable NaN over the other's files. Read every variable
    # only from the files that actually contain it.
    VAR_FILE_GLOB = {'mx2t': '*min-max-temp*.nc',
                     'mn2t': '*min-max-temp*.nc',
                     'tp': '*total-prep*.nc'}

    # ERA5 total precipitation is hourly accumulation in metres; store daily mm.
    VAR_SCALE = {'tp': 1000.0}

    def __init__(self, transform, data_path="../data/era5/*.nc",
                 surf_vars=('mx2t', 'mn2t'),
                 lat_lim=None, lon_lim=None,
                 split_val=False,
                 val_years=(2010, 2015,2023),
                 return_time_period=False,
                 deseason_standardize=False):
        self.transform = transform
        self.return_time_period = return_time_period
        self.deseason_standardize = deseason_standardize
        self.imgs, self.validation_imgs = self.load_images(
            data_path, list(surf_vars), lat_lim, lon_lim,
            split_val=split_val, val_years=tuple(val_years),
        )

    def _resolve_var_files(self, data_path, surf_vars):
        """Map each variable to the sorted list of files that hold it."""
        import glob
        root = os.path.dirname(data_path)
        var_files = {}
        for var in surf_vars:
            pattern = self.VAR_FILE_GLOB.get(var)
            path = os.path.join(root, pattern) if pattern else data_path
            files = sorted(glob.glob(path))
            if not files:
                raise FileNotFoundError(f'no ERA5 files for {var!r} matching {path}')
            var_files[var] = files
        return var_files

    def _daily_cache_path(self, var_files, surf_vars, lat_lim, lon_lim):
        """Deterministic cache file for the resampled daily tensor.

        Reading + hourly->daily resampling the raw netCDF (tens of GB) is by far
        the slowest step and it is fully determined by the input files and the
        spatial crop, so we cache the result keyed on a hash of those inputs.
        """
        import hashlib
        files = sorted({f for fs in var_files.values() for f in fs})
        sig = [(os.path.basename(f), os.path.getmtime(f), os.path.getsize(f)) for f in files]
        sig.append((tuple(surf_vars), tuple(lat_lim) if lat_lim else None,
                    tuple(lon_lim) if lon_lim else None))
        key = hashlib.md5(repr(sig).encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(files[0]) if files else '.', '.daily_cache')
        return os.path.join(cache_dir, f'daily_{key}.pt')

    def _crop(self, ds, lat_lim, lon_lim):
        if lat_lim is None or lon_lim is None:
            return ds
        lat_min, lat_max = min(lat_lim), max(lat_lim)
        lon_min, lon_max = min(lon_lim), max(lon_lim)
        # ERA5 latitude is typically stored in descending order.
        lat_descending = ds.latitude.values[0] > ds.latitude.values[-1]
        lat_slice = slice(lat_max, lat_min) if lat_descending else slice(lat_min, lat_max)
        return ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

    def load_images(self, data_path, surf_vars, lat_lim, lon_lim,
                    split_val, val_years):
        var_files = self._resolve_var_files(data_path, surf_vars)
        cache_path = self._daily_cache_path(var_files, surf_vars, lat_lim, lon_lim)
        if os.path.exists(cache_path):
            blob = torch.load(cache_path, weights_only=False)
            x_surf, times_index = blob['x_surf'], blob['times_index']
            logger.info(f'loaded cached daily tensor from {cache_path}')
        else:
            import xarray as xr  # local import: keep optional at module level

            # Group the variables that share a file set so each set is opened once.
            groups = {}
            for var in surf_vars:
                groups.setdefault(tuple(var_files[var]), []).append(var)

            daily = {}  # var -> (DatetimeIndex, tensor of shape (T, H, W))
            for files, group_vars in groups.items():
                ds = xr.open_mfdataset(list(files), concat_dim="valid_time", combine='nested')
                ds = self._crop(ds, lat_lim, lon_lim)

                for var in group_vars:
                    agg = self.DAILY_AGG.get(var, 'mean')
                    da_daily = getattr(ds[var].resample(valid_time='1D'), agg)()
                    values = torch.from_numpy(da_daily.values).float()
                    values *= self.VAR_SCALE.get(var, 1.0)
                    daily[var] = (pd.to_datetime(da_daily['valid_time'].values), values)
                    logger.info(f'resampled {var} to daily ({agg}): {tuple(values.shape)}')

            # Variables may cover slightly different periods, so keep the days
            # every variable has and stack them in the configured order.
            times_index = daily[surf_vars[0]][0]
            for var in surf_vars[1:]:
                times_index = times_index.intersection(daily[var][0])
            times_index = times_index.sort_values()

            daily_arrays = []
            for var in surf_vars:
                var_times, values = daily[var]
                daily_arrays.append(values[var_times.get_indexer(times_index)])

            x_surf = torch.stack(daily_arrays, dim=1)  # (T, V, H, W)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({'x_surf': x_surf, 'times_index': times_index}, cache_path)
            logger.info(f'saved daily tensor cache to {cache_path}')

        if split_val:
            val_mask = times_index.year.isin(list(val_years))
            val_indices = np.where(val_mask)[0]
            train_indices = np.where(~val_mask)[0]

            validation = x_surf[val_indices]
            x_surf = x_surf[train_indices]

            self.time = times_index[train_indices]
            self.val_time = times_index[val_indices]
        else:
            validation = None
            self.time = times_index

        if self.deseason_standardize:
            # Per-calendar-month, per-grid-cell z-score: deseasonalises (removes the
            # annual cycle) and standardises in one step. Climatology (mean/std) is
            # estimated on TRAIN days only, then applied to both train and val.
            mu, sigma = self._monthly_cell_climatology(x_surf, self.time)
            x_surf = self._apply_monthly_cell_zscore(x_surf, self.time, mu, sigma)
            if validation is not None:
                validation = self._apply_monthly_cell_zscore(validation, self.val_time, mu, sigma)
            logger.info('applied per-month per-cell deseasonalised standardisation (anomalies)')

        return x_surf, validation

    @staticmethod
    def _monthly_cell_climatology(x, times, eps=1e-6):
        """Return (mu, sigma) tensors of shape (12, V, H, W): per calendar-month,
        per-cell mean and std estimated over `times`."""
        months = torch.as_tensor(pd.DatetimeIndex(times).month.values)
        V, H, W = x.shape[1:]
        mu = torch.zeros(12, V, H, W)
        sigma = torch.ones(12, V, H, W)
        for m in range(1, 13):
            sel = months == m
            if sel.sum() == 0:
                continue
            xm = x[sel]
            mu[m - 1] = xm.mean(0)
            sigma[m - 1] = xm.std(0, unbiased=False).clamp_min(eps)
        return mu, sigma

    @staticmethod
    def _apply_monthly_cell_zscore(x, times, mu, sigma):
        months = torch.as_tensor(pd.DatetimeIndex(times).month.values) - 1
        return (x - mu[months]) / sigma[months]

    def __getitem__(self, index):
        cimg = self.imgs[index]
        timg = self.transform(cimg) if self.transform is not None else cimg

        if self.return_time_period:
            period = self.time[index].to_period('D')
            ordinal = period.ordinal if isinstance(period, pd.Period) else period.asi8
            return timg, torch.tensor(ordinal)
        return timg, torch.tensor(-1)

    def __len__(self):
        return self.imgs.shape[0]
