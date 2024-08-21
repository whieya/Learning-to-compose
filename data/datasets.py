from __future__ import annotations
import random
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from PIL import Image
import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import center_crop

from data.dataset_filtering import FilterStringParser
from data.dataset_transform import DatasetTransform, get_dataset_transform
from data.types import DataDict, FeatureMetadata, MetadataDict
# from utils.paths import CONFIG, DATA
from utils.utils import rename_dict_keys

from torchvision import transforms
from einops import rearrange

import json
from pycocotools import mask as coco_mask

_DUMMY_DATA_SIZE = 10000


@dataclass
class MultiObjectDataset(Dataset):
    """
    Base class for multi-object datasets.
    Borrowed from ihttps://github.com/addtt/object-centric-library.git 
    """ 

    name: str
    width: int
    height: int
    max_num_objects: int
    num_background_objects: int
    input_channels: int
    dataset_size: int
    dataset_path: str
    dataset_fname: str  

    # Features to be returned when loading the dataset. If None, returns all features (e.g. including masks).
    output_features: Union[Literal["all"], List[str]]
    variant: Optional[str] = None
    starting_index: int = 0
    transform: transforms = None

    # Define the dataset by filtering an existing dataset loaded from disk.
    dataset_filter_string: Optional[str] = None

    # Filter string to define the held out set
    heldout_filter_string: Optional[str] = None

    # Whether to retrieve the heldout version of the dataset or not (negating filter string).
    heldout_dataset: bool = False

    dataset_transform: Optional[str] = None
    dataset_transform_op: DatasetTransform = field(init=False)

    # Skip loading actual data, use fake data instead.
    skip_loading: bool = False

    # Callable to postprocess entire sample at the end of `__getitem__`.
    postprocess_sample: Optional[Callable[[Dict, MultiObjectDataset], Dict]] = None

    # Additional dataset filtering with a callable that takes this dataset as input.
    callable_filter: Optional[Callable[[MultiObjectDataset], np.ndarray]] = None

    def __post_init__(self):
        super().__init__()
        # self._check_args()

        if self.variant is not None:
            self.identifier = self.name + "-" + self.variant
        else:
            self.identifier = self.name
        self.full_dataset_path = Path(self.dataset_path) / self.dataset_fname

        if self.skip_loading:
            logging.info("skip_loading is True: dummy data will be used")
            self.dataset, self.metadata = self._load_dummy_data()
            self.downstream_features = []
        else:
            self.dataset, self.metadata = self._load_data()
        self.data = {}

        # From filter strings to mask.
        self.mask = self._compute_filter_mask()  # boolean (N,)

        # Compute ranges to load contiguously in RAM. Faster loading when `mask` is sparse.
        self.preload_range, self.idx_range = _minimal_load_range(
            starting_index=self.starting_index,
            dataset_size=self.dataset_size,
            mask=self.mask,
        )

        # Load necessary subset of data.
        if self.output_features == "all":
            self.output_features = list(self.dataset.keys())
        for feature in self.output_features:
            self.data[feature] = self.dataset[feature][
                self.preload_range[0] : self.preload_range[1]
            ][self.idx_range]

        # Fix object indices in Objects Room at loading time.
        self._fix_objects_room_labels()

        # Picks the required dataset transform and instantiates it with this dataset.
        # When created, the transform object modifies data and metadata of this dataset.
        self.dataset_transform_op: DatasetTransform = get_dataset_transform(self)

        # self.downstream_metadata = self._get_downstream_metadata(
        #     self.downstream_features
        # )

        # self.features_size = sum(
        #     metadata.slice.stop - metadata.slice.start
        #     for metadata in self.downstream_metadata
        # )

        # Delete dataset because it is not used anymore after init, and it breaks data
        # loading when num_workers>0 (it contains HDF5 objects which cannot be pickled).
        del self.dataset

    def _check_args(self):
        assert isinstance(self.name, str)
        assert isinstance(self.width, int)
        assert isinstance(self.height, int)
        assert isinstance(self.max_num_objects, int)
        assert isinstance(self.num_background_objects, int)
        assert isinstance(self.input_channels, int)
        assert isinstance(self.dataset_size, int)
        assert isinstance(self.dataset_fname, str)
        assert (
            self.output_features == "all"
            or isinstance(self.output_features, list)
            and all(isinstance(x, str) for x in self.output_features)
        )
        assert self.variant is None or isinstance(self.variant, str)
        assert isinstance(self.starting_index, int)
        assert self.dataset_filter_string is None or isinstance(
            self.dataset_filter_string, str
        )
        assert self.heldout_filter_string is None or isinstance(
            self.heldout_filter_string, str
        )
        assert isinstance(self.heldout_dataset, bool)
        assert isinstance(self.downstream_features, (list, ListConfig)) and all(
            isinstance(x, str) for x in self.downstream_features
        )
        assert self.dataset_transform is None or isinstance(self.dataset_transform, str)
        assert isinstance(self.skip_loading, bool)

    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        """Loads data and metadata.

        By default, the data is a dict with h5py.Dataset values, but when overriding
        this method we allow arrays too."""
        return _load_data_hdf5(data_path=self.full_dataset_path)

    def _load_dummy_data(self) -> Tuple[Dict[str, np.ndarray], MetadataDict]:
        """Loads dummy data for testing.

        Returns:
            tuple containing data and metadata.
        """
        data = {
            "image": np.random.rand(
                _DUMMY_DATA_SIZE, self.height, self.width, self.input_channels
            ),
            "mask": np.zeros([_DUMMY_DATA_SIZE, self.height, self.width, 1]),
            "num_actual_objects": np.ones([_DUMMY_DATA_SIZE, 1]),
            "visibility": np.ones([_DUMMY_DATA_SIZE, self.max_num_objects]),
        }
        metadata = {
            "self.dataset": {
                "type": "dataset_property",
                "num_samples": _DUMMY_DATA_SIZE,
            },
            "visibility": {
                "type": "categorical",
                "num_categories": 1,  # other fields are not used
            },
        }
        return data, metadata

    def __len__(self):
        return len(self.idx_range)

    def _preprocess_feature(self, feature: np.ndarray, feature_name: str) -> Any:
        """Preprocesses a dataset feature at the beginning of `__getitem__()`.

        Args:
            feature: Feature data.
            feature_name: Feature name.

        Returns:
            The preprocessed feature data.
        """
        if feature_name == "image":
            return (
                torch.as_tensor(feature, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )
        if feature_name == "mask":
            one_hot_masks = F.one_hot(
                torch.as_tensor(feature, dtype=torch.int64),
                num_classes=self.max_num_objects,
            )
            # (num_objects, 1, height, width)
            return one_hot_masks.permute(3, 2, 0, 1).to(torch.float32)
        if feature_name == "visibility":
            feature = torch.as_tensor(feature, dtype=torch.float32)
            if feature.dim() == 1:  # e.g. in ObjectsRoom
                feature.unsqueeze_(1)
            return feature
        if feature_name == "num_actual_objects":
            return torch.as_tensor(feature, dtype=torch.float32)
        if feature_name in self.metadata.keys():
            # Type is numerical, categorical, or dataset_property.
            feature_type = self.metadata[feature_name]["type"]
            if feature_type == "numerical":
                return _normalize_numerical_feature(
                    feature, self.metadata, feature_name
                )
            if feature_type == "categorical":
                return _onehot_categorical_feature(
                    feature, self.metadata[feature_name]["num_categories"]
                )
        return feature

    def __getitem__(self, idx):
        out = {}

        out['image'] = self.transform(
                self._preprocess_feature(
                    self.data['image'][idx], 'image')
        )

        for feature_name in self.data.keys():
            if feature_name != 'image':
                out[feature_name] = self._preprocess_feature(
                    self.data[feature_name][idx], feature_name
                )

        out = self.dataset_transform_op(out, idx)

        out["is_foreground"] = out["visibility"].clone()
        out["is_foreground"][: self.num_background_objects] = 0.0
        out["sample_id"] = self._get_raw_idx(idx)

        # Per-object variable indicating whether an object was modified by a transform.
        if "is_modified" not in out:
            out["is_modified"] = torch.zeros_like(out["visibility"]).squeeze()
        else:  # TODO fix type of is_modified so this is not necessary
            out["is_modified"] = torch.FloatTensor(out["is_modified"])

        if self.postprocess_sample is not None:
            out = self.postprocess_sample(out, self)

        assert out["is_modified"].dtype == torch.float32, out["is_modified"].dtype
        assert out["visibility"].shape == (self.max_num_objects, 1)
        assert out["mask"].shape == (self.max_num_objects, 1, self.height, self.width)
        assert out["mask"].sum(1).max() <= 1.0
        assert out["mask"].min() >= 0.0

        return out

    def _get_raw_idx(self, idx):
        return self.preload_range[0] + self.idx_range[idx]

    def _get_downstream_metadata(
        self, feature_names: Optional[List[str]] = None, sort_features: bool = True
    ) -> List[FeatureMetadata]:
        """Returns the metadata for features to be used in downstream tasks.

        Args:
            feature_names (list): List of feature names for downstream tasks.
            sort_features (bool): if True, the list of features will be sorted
                according to the standard order specified in `_feature_index`.

        Returns:
            List of `FeatureMetadata`, which contains the location of each feature
            in the overall feature array, the type of the feature (numerical or
            categorical), and its name.

        """
        if feature_names is None:
            return []
        if sort_features:
            feature_names = sorted(feature_names, key=_feature_index)
        feature_infos = []
        start_index = 0
        for feature_name in feature_names:
            metadata = self.metadata[feature_name]
            if metadata["type"] == "categorical":
                length_feature = int(metadata["num_categories"])
            elif metadata["type"] == "numerical":
                length_feature = int(metadata["shape"][-1])
            else:
                raise ValueError(
                    "Metadata type '{}' not recognized.".format(metadata["type"])
                )
            feature_infos.append(
                FeatureMetadata(
                    feature_name,
                    metadata["type"],
                    slice(start_index, start_index + length_feature),
                )
            )
            start_index += length_feature
        return feature_infos

    def _compute_filter_mask(self) -> np.ndarray:
        """Returns the mask for filtering the dataset according to the filters currently in place.

        The mask is an AND of:
        1. the standard filter string `dataset_filter_string`,
        2. the filter string defining heldout experiments `heldout_filter_string`,
        3. the additional `callable_filter` that returns a mask.

        Returns:
            The boolean mask to be applied to the dataset (on the first dimension).
        """

        # The metadata key `self.dataset` contains information regarding the dataset
        # itself. It is meta information that does not pertain to any feature of the
        # dataset in particular.
        full_size_dataset = self.metadata["self.dataset"]["num_samples"]

        parser = FilterStringParser(self.dataset, self.num_background_objects)

        # Process the standard filter string.
        if self.dataset_filter_string is None:
            dataset_mask = np.ones((full_size_dataset,), dtype=bool)
        else:
            dataset_mask = parser.filter_string_to_mask(self.dataset_filter_string)

        # Process the heldout filter string.
        if self.heldout_filter_string is None:
            heldout_mask = np.ones((full_size_dataset,), dtype=bool)
        else:
            heldout_mask = parser.filter_string_to_mask(self.heldout_filter_string)

            # If we do not want the heldout part, negate the mask.
            # Assume we checked that `heldout_dataset` is a bool.
            if not self.heldout_dataset:
                heldout_mask = ~parser.filter_string_to_mask(self.heldout_filter_string)

        # Additional filtering with a callable that takes this dataset as input.
        if self.callable_filter is not None:
            logging.info("Masking dataset with provided callable_filter")
            dataset_mask &= self.callable_filter(self)

        # Return the intersection of the masks.
        return heldout_mask & dataset_mask

    def _fix_objects_room_labels(self):
        """Fixes slot numbers in Objects Room.

        The number of background objects in Objects Room is variable. This method
        attempts to identify background objects from their masks and, if they are
        less than 4, it shifts foreground objects such that they occupy slots with
        index larger than 3. The idea is that the first 4 slots are always background.
        From visual inspection of a few hundred images, it appears to be accurate.
        Doing this at runtime allows to adjust this method in the future while
        retaining the original dataset.
        """
        if len(self.output_features) > 0 and self.name == "objects_room":
            for i in range(len(self)):
                update = False
                m = self.data["mask"][i]

                last_bgr_id = self.num_background_objects - 1
                left_col = m[:, 0, 0]
                right_col = m[:, m.shape[1] - 1, 0]
                left_size = left_col[left_col == last_bgr_id].size
                right_size = right_col[right_col == last_bgr_id].size
                on_left_border = left_size > 0
                on_right_border = right_size > 0

                min_pixels = 3
                last_bgr_mask = m == last_bgr_id
                wall_mask = m == last_bgr_id - 1
                floor_sky_mask = m < 2
                #
                border_above_mask = wall_mask[:-1] & last_bgr_mask[1:]
                border_cols = np.where(border_above_mask)[1]
                borders_above_wall = (
                    len(border_cols) > 0
                    and border_cols.max() - border_cols.min() + 1 >= min_pixels
                )
                #
                border_below_mask = wall_mask[1:] & last_bgr_mask[:-1]
                border_cols = np.where(border_below_mask)[1]
                borders_below_wall = (
                    len(border_cols) > 0
                    and border_cols.max() - border_cols.min() + 1 >= min_pixels
                )
                #
                border_right_mask = last_bgr_mask[:, :-1] & floor_sky_mask[:, 1:]
                border_rows = np.where(border_right_mask)[0]
                borders_right_floor_sky = (
                    len(border_rows) > 0
                    and border_rows.max() - border_rows.min() + 1 >= min_pixels
                )
                #
                border_left_mask = last_bgr_mask[:, 1:] & floor_sky_mask[:, :-1]
                border_rows = np.where(border_left_mask)[0]
                borders_left_floor_sky = (
                    len(border_rows) > 0
                    and border_rows.max() - border_rows.min() + 1 >= min_pixels
                )

                # If no apparent foreground objects, assume there are only N-k bgr
                # objects, and move the last k slots by k positions.
                if self.data["num_actual_objects"][i].item() <= 0:
                    # plt.imshow(m.astype(int), cmap="tab10")
                    # plt.colorbar()
                    # plt.savefig(f"tmp_images/_few-objects_{i}")
                    # plt.close()
                    m[m == m.max()] = self.num_background_objects
                    update = True

                # Assume there are either 3 or 4 bgr objects. Then if the following
                # conditions hold, it's probably not background, and we can shift it
                # up by 1 in the mask, along with all objects with higher indices.
                elif m.max() < self.max_num_objects - 1 and (
                    borders_above_wall
                    or borders_below_wall
                    or (borders_right_floor_sky and borders_left_floor_sky)
                    or (borders_right_floor_sky and not on_left_border)
                    or (borders_left_floor_sky and not on_right_border)
                ):
                    # plt.imshow(m.astype(int), cmap="tab10")
                    # plt.colorbar()
                    # plt.savefig(f"tmp_images/{i}")
                    # plt.close()
                    m[m >= self.num_background_objects - 1] += 1
                    update = True

                # Update visibility and num actual objects
                if update:
                    self.data["visibility"][i].fill(0)
                    self.data["visibility"][i][np.unique(m)] = 1
                    self.data["num_actual_objects"][i] = (
                        self.data["visibility"][i].sum() - self.num_background_objects
                    )
                # else:
                #     plt.imshow(m.astype(int), cmap="tab10")
                #     plt.colorbar()
                #     plt.savefig(f"ignored_images/{i}")
                #     plt.close()


class Clevr(MultiObjectDataset):
    def _load_data(self) -> Tuple[DataDict, MetadataDict]:
        data, metadata = super()._load_data()

        # 'pixel_coords' shape: (B, num objects, 3)
        data["x_2d"] = data["pixel_coords"][:, :, 0]
        data["y_2d"] = data["pixel_coords"][:, :, 1]
        data["z_2d"] = data["pixel_coords"][:, :, 2]
        del data["pixel_coords"]
        del metadata["pixel_coords"]
        return data, metadata


def make_dataset(
        dataset_name: str, dataset_path: str, starting_index: int, dataset_size: int, flip: bool=False) -> MultiObjectDataset:
    print(
        f"Instantiating dataset with starting_index={starting_index} and size={dataset_size}."
    )
    print(f"Dataset config:\n{dataset_name}")

    augmentations = transforms.Normalize([0.5], [0.5])
    assert dataset_name == 'clevr' or dataset_name == 'clevrtex'
    if dataset_name == 'clevr':
        dataset_fname = f'{dataset_name}_10-full.hdf5'
    else:
        dataset_fname = f'{dataset_name}-full.hdf5'

    dataset = Clevr(
            name=dataset_name,
            width=128,
            height=128,
            max_num_objects=11,
            num_background_objects=1,
            input_channels=3,
            starting_index=starting_index,
            dataset_size=dataset_size,
            dataset_path=dataset_path,
            dataset_fname=dataset_fname,
            output_features='all',
            transform = augmentations)
    return dataset

def make_dataloaders(
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    data_sizes: Optional[List[int]] = None,
    shuffle: bool = True,
    starting_index: int = 0,
    pin_memory: bool = True,
    num_workers: int = 0,
    eval_mode: bool=False,
    steps: int=0,
    return_properties: bool=False,
) -> List[DataLoader]:
    """Generates a list of dataloaders.

    The size of each dataloader is given by `data_sizes`.

    Args:
        dataset_config: the config for the dataset from which data is selected.
        batch_size: batch size for all dataloaders.
        data_sizes: a list of ints with sizes of each data split.
        starting_index:
        pin_memory:
        num_workers:

    Returns:
        List of dataloaders
    """
    if data_sizes is None:
        return []
    dataloaders = []
    start = starting_index

    dataloaders = []
    start = starting_index

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    if dataset_name == 'clevr' or 'clevrtex' in dataset_name:
        splits = ['train', 'val', 'test']
        for idx, size in enumerate(data_sizes):
            _shuffle = True if idx==0 else False
            dataloaders.append(
                make_dataloader(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    dataset_size=size,
                    starting_index=start,
                    shuffle=_shuffle and not eval_mode,
                    drop_last=_shuffle and not eval_mode,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    steps=steps,
                    split=splits[idx],
                )
            )
            start += size
    else:
        if 'ptr' in dataset_name:
            num_segs = 6+1
            target_dataset = PTR
        elif 'msn-easy' in dataset_name:
            num_segs = 4+1
            target_dataset = MSN_Easy
        else:
            raise ValueError(f'Unknown dataset name {dataset_name}')

        train_dataset = target_dataset(
            root=dataset_path,
            split='train',
            img_size=128,
            num_segs=num_segs,
            return_properties=return_properties,
            )

        val_dataset = target_dataset(
            root=dataset_path,
            split='val',
            img_size=128,
            num_segs=num_segs,
            return_properties=return_properties,
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
            )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
            )

        dataloaders = [train_dataloader, val_dataloader, None]
    return dataloaders


def make_dataloader(
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    dataset_size: int,
    starting_index: int = 0,
    shuffle=False,
    drop_last=False,
    pin_memory: bool = True,
    num_workers: int = 0,
    flip: bool=False,
    steps: int = 0,
    split: str = 'train',
) -> DataLoader:
    dataset = make_dataset(dataset_name, dataset_path, starting_index, dataset_size, flip=False)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )



def _normalize_numerical_feature(
    data: np.array, metadata: MetadataDict, feature_name: str
) -> Tensor:
    mean = metadata[feature_name]["mean"].astype("float32")
    std = np.sqrt(metadata[feature_name]["var"]).astype("float32")
    return torch.as_tensor((data - mean) / (std + 1e-6), dtype=torch.float32)


def _onehot_categorical_feature(data: np.array, num_classes: int) -> Tensor:
    tensor = torch.as_tensor(data, dtype=torch.int64).squeeze(-1)
    return F.one_hot(tensor, num_classes=num_classes).to(torch.float32)


def _load_data_hdf5(
    data_path: Path, metadata_suffix: str = "metadata.npy"
) -> Tuple[Dict[str, h5py.Dataset], MetadataDict]:
    """Loads data and metadata assuming the data is hdf5, and converts it to dict."""
    metadata_fname = f"{data_path.stem.split('-')[0]}-{metadata_suffix}"
    metadata_path = data_path.parent / metadata_fname
    metadata = np.load(str(metadata_path), allow_pickle=True).item()
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Metadata type {type(metadata)}, expected instance of dict")
    dataset = h5py.File(data_path, "r")
    # From `h5py.File` to a dict of `h5py.Datasets`.
    dataset = {k: dataset[k] for k in dataset}
    return dataset, metadata


def _minimal_load_range(
    starting_index: int, dataset_size: int, mask: np.ndarray
) -> Tuple[Tuple[int, int], np.ndarray]:
    start_idx = starting_index
    end_idx = starting_index + dataset_size
    masked_indices = np.arange(len(mask))[mask]
    idx_range = masked_indices[start_idx:end_idx]  # shape (dataset_size,)
    info = (
        f"There are {len(mask)} samples before masking, {len(masked_indices)} after "
        f"masking, and the required starting index (after masking) is {start_idx}."
    )
    logging.info(info)
    if dataset_size > len(idx_range):
        raise ValueError(
            f"Required dataset size is {dataset_size} but only {len(idx_range)} samples available. {info}"
        )
    return (min(idx_range), max(idx_range) + 1), idx_range - min(idx_range)


def _feature_index(x: str) -> int:
    """Returns the index of the given feature in the pre-defined canonical order."""
    feature_order = ["size", "scale", "material", "shape", "x", "y", "z", "color"]
    if x in feature_order:
        return feature_order.index(x)
    else:
        return len(feature_order)


def get_available_dataset_configs() -> List[str]:
    """Returns the (sorted) names of the datasets for which a YAML config is available."""
    
    # data_path = CONFIG / "dataset"
    # out = []
    # for file in data_path.iterdir():
    #    if file.is_dir() or file.suffix != ".yaml":
    #         continue
    #     out.append(file.stem)

    # modified to download only clevrtex or clevr
    out = ['clevrtex', 'clevr']
    return sorted(out)

class MSN_Easy(Dataset):
    def __init__(self, root, img_size, num_segs, split='train', mode='train', return_properties=False):
        self.split=split
        self.root = os.path.join(root, split)

        self.img_size = img_size
        self.transform = self._get_transforms()
       
        self.total_imgs = [x for x in Path(self.root).glob('*/*.png') if 'image' in x.name]
        self.total_imgs = sorted(self.total_imgs)

        self.num_segs = num_segs
        print(self.root)
        print(f'length of the dataset : {len(self.total_imgs)}')

    def _get_transforms(self):
        augmentations = transforms.Compose(
                [transforms.CenterCrop(240),
                transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        return augmentations


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img = Image.open(self.total_imgs[idx]).convert('RGB')
        img = self.transform(img)

        if self.split=='train':
            return {'image':img}

        elif self.split == 'val':
            # load mask
            masks_path = str(self.total_imgs[idx]).replace('image', 'mask')
            masks_path = masks_path.replace('png', 'npy')
            masks = np.load(masks_path)

            mask = []
            for i in range(self.num_segs):
                # shape of mask : (height, width)
                # normalize to one-hot
                _mask = torch.as_tensor(masks==(i+1), dtype=torch.long).squeeze(-1)
                
                # resize to 128x128 masks
                resized_mask = torch.nn.functional.interpolate(_mask[:, 40:-40].float().unsqueeze(0).unsqueeze(0), size=(128,128)).long().squeeze()

                mask.append(resized_mask)
    
            # stack mask : shape of mask (num_segs, height, width)
            mask = torch.stack(mask, dim=0)
            
            is_foreground = (mask.sum(-1).sum(-1) > 0).long().unsqueeze(-1)
            is_foreground[0] = 0
            is_background = 1-is_foreground
            visibility = is_foreground
            is_modified = torch.zeros_like(visibility).squeeze(-1)

            # align the format : shape is now become (num_segs, 1, height, width)
            mask = mask.unsqueeze(1)
            return {'image':img, 'mask':mask, 'is_foreground':is_foreground,
                    'is_background':is_background,
                    'is_modified':is_modified,
                    'visibility':visibility,
                    }


class PTR(Dataset):
    def __init__(self, root, img_size, num_segs, split='train', return_properties=False):
        self.split=split
        self.root = os.path.join(root, split)
        print(self.root)
        self.img_size = img_size
        self.total_imgs = sorted([str(x) for x in Path(self.root).glob('images/*.png')])
        self.num_segs = num_segs
        self.transform = self._get_transforms()
        self.height, self.width = 600, 800
        self.return_properties = return_properties

        self.cate2idx = {
            'Chair': 0,
            'Table': 1,
            'Bed': 2,
            'Refrigerator': 3,
            'Cart': 4,
        }
        print(f'length of the dataset : {len(self.total_imgs)}')

    def _get_transforms(self):
        augmentations = transforms.Compose(
                [transforms.CenterCrop(600),
                transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        return augmentations


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_path = self.total_imgs[idx]
        img  = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        mask_path = img_path.replace('images', 'scenes').replace('png', 'json')

        if self.split=='train' and not self.return_properties:
            return {'image':img}

        elif self.split=='val' or self.return_properties:
            masks = torch.zeros((self.num_segs, self.img_size, self.img_size))
            xs = torch.zeros((self.num_segs))
            ys = torch.zeros((self.num_segs))
            categories = -torch.ones((self.num_segs)).long()
            with open(mask_path, 'r') as json_file:
                data = json.load(json_file)
                num_objs = len(data['objects'])
                for i in range(1, num_objs+1):
                    rle_mask = data['objects'][i-1]['obj_mask']
                    try:
                        mask = coco_mask.decode(rle_mask)
                    except TypeError:
                        print('error', mask_path, i)
                    mask = torch.as_tensor(mask, dtype=torch.long)
                    mask = torch.nn.functional.interpolate(mask[:, 100:-100].float().unsqueeze(0).unsqueeze(0), size=(self.img_size, self.img_size)).long().squeeze()
                    masks[i] = mask
                    xs[i] = data['objects'][i-1]['pixel_coords'][0]
                    ys[i] = data['objects'][i-1]['pixel_coords'][1]
                    categories[i] = self.cate2idx[data['objects'][i-1]['category']]
    
            # stack mask : shape of mask (num_segs, height, width)
            is_foreground = (masks.sum(-1).sum(-1) > 0).long().unsqueeze(-1)
            is_foreground[0] = 0
            is_background = 1-is_foreground
            visibility = is_foreground
            is_modified = torch.zeros_like(visibility).squeeze(-1)

            # align the format : shape is now become (num_segs, 1, height, width)
            masks = masks.unsqueeze(1)
            return {'image':img, 'mask':masks, 'is_foreground':is_foreground,
                    'is_background':is_background,
                    'is_modified':is_modified,
                    'visibility':visibility,
                    'x': xs,
                    'y': ys,
                    'category': categories,
                    }

