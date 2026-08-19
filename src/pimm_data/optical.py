"""
OpticalDataset — PMT light-readout dataset for goop optical output.

A single-modality sibling of :class:`~pimm_data.jaxtpc.JAXTPCDataset` /
:class:`~pimm_data.lucid.LUCiDDataset`. It composes one optical reader over
sharded ``sensor/`` files and emits a **nested** dict with one ``sensor``
modality of per-**chunk** waveform rows::

    {
      'sensor': {
        'pmt_id':   (K,)   int32    — channel (per-side for east/west)
        't0_ns':    (K,)   float32  — absolute chunk start time
        'length':   (K,)   int32    — samples per chunk
        'pe':       (K, 1) float32  — channel PE total within the group
        'instance': (K,)   int32    — group id (interaction for label, side for e/w)
        'adc':      (ΣL,)  float32  — packed pedestal-subtracted waveforms
        '_roles':   {'adc': ('instance', 'sensor_wave_offset')},
      },
      'name': str, 'split': str,
    }

Two goop schemas are selectable via ``schema=`` (see
:mod:`pimm_data.readers.optical_sensor`):

* ``'label'`` (default) — ``label_K`` groups; ``instance`` = interaction.
* ``'east_west'`` — ``{east,west}`` groups; ``instance`` = side; files globbed
  as ``*.h5`` (e.g. ``light_output.h5``).

Each **chunk** is one row of the ``sensor`` part (``sensor_offset`` counts chunks
/ event); the raw samples are a packed SECOND row-space (``sensor_wave_offset``
counts samples / event), tagged ``('instance','sensor_wave_offset')`` so collate
concats them and ``split_event`` slices by sample count (REDESIGN §3). No ``labl``
join; per-interaction TPC truth (``tpc_*``) is left on disk.

Canonical pipeline (note the second offset + per-chunk ``length`` to recover
chunk boundaries in the packed ``adc``)::

    OpticalDataset(data_root=..., dataset_name='...', schema='label',
      transform=[dict(type='Collect', parts={
          'sensor': dict(
              keys=('pmt_id', 't0_ns', 'length', 'pe', 'instance', 'adc'),
              feat_keys=('pe',),
              offset_keys_dict=dict(offset='pmt_id', wave_offset='adc'),
          )})])

Registered in :data:`pimm_data.DATASETS`.
"""

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from .readers.optical_sensor import OpticalSensorReader, OpticalEastWestReader


@DATASETS.register_module()
class OpticalDataset(ShardEventDataset):
    """PMT light dataset of per-chunk waveforms (one modality, two schemas)."""

    #: Only the optical readout is loaded today (TPC truth stays on disk).
    VALID_MODALITIES = ('sensor',)

    _READERS = {'label': OpticalSensorReader, 'east_west': OpticalEastWestReader}

    def __init__(
        self,
        data_root,
        split='',
        modalities=('sensor',),
        dataset_name='optical',
        schema='label',
        decode_digitization=True,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        if schema not in self._READERS:
            raise ValueError(
                f"schema={schema!r} not in {sorted(self._READERS)}")

        self._dataset_name = dataset_name
        self._schema = schema
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.sensor_reader = None
        if 'sensor' in self._modalities:
            self.sensor_reader = self._READERS[schema](
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name,
                decode_digitization=decode_digitization)

        self._canonical_reader = self.sensor_reader
        self._build_joint_index(source_label=f"OpticalDataset({data_root!r})")

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )

    def get_data(self, idx):
        # data_list VALUES are event ids. Identical to `idx % len(...)` while
        # the list is range(n), and the only correct form once it is not.
        real_idx = self.data_list[idx % len(self.data_list)]
        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }
        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor(
                self.sensor_reader.read_event(real_idx))
        return data

    def _build_sensor(self, raw):
        """Per-chunk light cloud: clean named columns + packed ``adc`` as the
        second (sample) row-space; ``instance`` = group id."""
        return {
            'pmt_id': raw['pmt_id'],
            't0_ns': raw['t0_ns'],
            'length': raw['length'],
            'pe': raw['pe'][:, None].astype(np.float32),
            'instance': raw['instance'],
            'adc': raw['adc'],
            # adc is the packed sample row-space (REDESIGN §3); stamped here so
            # the Collect config only needs keys= + the wave_offset entry.
            '_roles': {'adc': ('instance', 'sensor_wave_offset')},
        }
