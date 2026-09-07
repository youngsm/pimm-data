"""pimm-data — multimodal detector dataset loaders + transform library.

Imports are ordered so that registering side-effects populate the
:data:`TRANSFORMS` and :data:`DATASETS` registries before anything
downstream looks them up.
"""

# Register blosc/zstd/lz4 HDF5 filters at import so production output
# written with those codecs (run_batch --codec, default blosc-zstd) is
# transparently readable. No-op if hdf5plugin isn't installed.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Registries + build function
from ._registry import Registry, build_from_cfg
from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS

# Collate utilities
from .collate import collate_fn, point_collate_fn, inseg_collate_fn

# Dataset classes (register themselves)
from .jaxtpc import JAXTPCDataset
from .lucid import LUCiDDataset
from .optical import OpticalDataset
from .pilarnet import PILArNetH5Dataset
from .multimodal import MultiModalEventDataset

# Detector-specific transforms (register PDGToSemantic)
from . import detector_transforms  # noqa: F401

# Torch-free public surface: HDF5 readers + the joint cross-modality index
# helper. This is the "bring-your-own-framework / roll-your-own-Dataset" path —
# the reader/index/decode layer imports only numpy/h5py (no torch), so a JAX or
# numpy consumer can read events without the torch transform layer. (Note: torch
# is still a required *install* dep — see ADR §5 — this exposes the framework-
# neutral reader code, not a torch-free import path.)
from .readers import (
    JAXTPCStepReader, JAXTPCSensorReader, JAXTPCHitsReader, JAXTPCLablReader,
    LUCiDStepReader, LUCiDSensorReader, LUCiDHitsReader, LUCiDLablReader,
    OpticalSensorReader, OpticalEastWestReader,
)
from ._joint_index import build_joint_index

# Plane-geometry registry (loaded from a JAXTPC-exported config JSON)
from .geometry import load_plane_registry, dataset_geometry_kwargs

# Flat-prefixed batch boundary helpers (REDESIGN)
from .batch_ops import to_batched_coords, split_event

# Post-collate, on-device batch transforms (the dense GPU path runner)
from . import dense_ops  # noqa: F401
from .batch_transforms import (
    sensor_dense_cfg,
    build_sensor_gpu_stages,
    move_to_device,
    content_seed,
    ToDevice,
    BatchTransformMixin,
)

__all__ = [
    "Registry",
    "build_from_cfg",
    "DATASETS",
    "build_dataset",
    "Compose",
    "TRANSFORMS",
    "collate_fn",
    "point_collate_fn",
    "inseg_collate_fn",
    "JAXTPCDataset",
    "LUCiDDataset",
    "OpticalDataset",
    "PILArNetH5Dataset",
    "MultiModalEventDataset",
    # readers + joint index (torch-free public surface)
    "JAXTPCStepReader",
    "JAXTPCSensorReader",
    "JAXTPCHitsReader",
    "JAXTPCLablReader",
    "LUCiDStepReader",
    "LUCiDSensorReader",
    "LUCiDHitsReader",
    "LUCiDLablReader",
    "OpticalSensorReader",
    "OpticalEastWestReader",
    "build_joint_index",
    # post-collate dense transforms (run via Compose; no batch-transform runner)
    "sensor_dense_cfg",
    "build_sensor_gpu_stages",
    "move_to_device",
    "content_seed",
    "ToDevice",
    "BatchTransformMixin",
    "load_plane_registry",
    "dataset_geometry_kwargs",
    # flat-prefixed batch boundary helpers (REDESIGN)
    "to_batched_coords",
    "split_event",
]

__version__ = "0.3.0"
