"""Readers for detector HDF5 files.

Each reader is self-contained: given a data_root and split, it discovers
shard files, builds an event index, and exposes ``read_event(idx)``
returning a flat ``dict[str, np.ndarray]``.
"""

from .jaxtpc_step import JAXTPCStepReader
from .jaxtpc_sensor import JAXTPCSensorReader
from .jaxtpc_labl import JAXTPCLablReader
from .jaxtpc_hits import JAXTPCHitsReader
from .lucid_step import LUCiDStepReader
from .lucid_sensor import LUCiDSensorReader
from .lucid_hits import LUCiDHitsReader
from .lucid_labl import LUCiDLablReader
from .optical_sensor import OpticalSensorReader, OpticalEastWestReader

__all__ = [
    "JAXTPCStepReader",
    "JAXTPCSensorReader",
    "JAXTPCLablReader",
    "JAXTPCHitsReader",
    "LUCiDStepReader",
    "LUCiDSensorReader",
    "LUCiDHitsReader",
    "LUCiDLablReader",
    "OpticalSensorReader",
    "OpticalEastWestReader",
]
