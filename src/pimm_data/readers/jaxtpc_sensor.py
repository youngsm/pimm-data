"""
JAXTPCSensorReader — reads sparse raw sensor signals from JAXTPC
``sensor/`` files. Supports both wire and pixel readouts.

Readout type is auto-detected from ``/config.readout_type`` and exposed
as ``reader.readout_type`` (``'wire'`` or ``'pixel'``). Output schema:

- wire:  ``sensor.{plane_label}.{wire, time, value}``
- pixel: ``sensor.{plane_label}.{py, pz, time, value}``

Handles both old format (planes directly under event) and new format
(planes under volume_N/ subgroups).
"""

import logging

import numpy as np
import h5py

from ._jaxtpc_readout import JAXTPCReadoutReader

# This module called log.warning() in an except handler without ever defining
# `log`, so a shard that failed num_time_steps detection raised NameError from
# inside the handler meant to survive it -- turning a warning into a crash on
# exactly the corrupt-input path. Both sibling readers (_base, _jaxtpc_readout)
# define their own; this one was missed.
log = logging.getLogger(__name__)


class JAXTPCSensorReader(JAXTPCReadoutReader):
    """Reads sparse raw sensor signals from JAXTPC ``sensor/`` HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing sensor shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_sensor_0000.h5').
    planes : str or list
        Which planes to load: 'all' or list like ['east_U', 'east_V'].
    decode_digitization : bool
        If True, subtract pedestal from uint16 values.
    n_wires_per_plane : dict or None
        Optional ``{plane_label: n_wires}`` fallback used when a plane group
        lacks an ``n_wires`` attr. The on-file attr, when present, wins. Surfaced
        (with ``num_time_steps``) so a downstream ``Densify`` transform can
        reconstruct a *fixed* ``(n_wires, n_ticks)`` grid.
    num_time_steps : int or None
        Optional override for the per-file tick count. When ``None`` it is read
        from ``/config`` (or the first event's) ``num_time_steps`` attr.
    pedestal_per_plane : dict or None
        Optional ``{plane_label: pedestal}`` fallback used when a plane group
        lacks a ``pedestal`` attr. The on-file attr, when present, wins. Surfaced
        so a downstream ``Digitize`` transform can quantize in raw-ADC space.
    """

    _MODALITY = 'sensor'

    def __init__(self, data_root, split='train', dataset_name='sim',
                 planes='all', decode_digitization=True,
                 n_wires_per_plane=None, num_time_steps=None,
                 pedestal_per_plane=None):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.planes = planes
        self.decode_digitization = decode_digitization
        # Geometry for densify: per-plane wire count + file-level tick count.
        # On-file attrs are authoritative; ctor kwargs are fallbacks.
        self.n_wires_per_plane = dict(n_wires_per_plane or {})
        # Per-plane pedestal (ADC) for a downstream Digitize; attr wins.
        self.pedestal_per_plane = dict(pedestal_per_plane or {})
        self._init_shards()
        self.readout_type = self._detect_readout_type()
        self.num_time_steps = (num_time_steps if num_time_steps is not None
                               else self._detect_num_time_steps())

    def _scan_readout_type(self, path):
        """Fallback scan for files without the readout_type attr. Handles old
        (planes directly under event) + new (planes under volume_N) layouts.
        Inspects only the first event group. Returns 'wire'/'pixel'/None."""
        with h5py.File(path, 'r', libver='latest', swmr=True) as f:
            for ek in f:
                if not ek.startswith('event_'):
                    continue
                evt = f[ek]
                for vk in evt:
                    vol = evt[vk]
                    if not isinstance(vol, h5py.Group):
                        continue
                    if vk.startswith('volume_'):
                        for pk in vol:
                            pg = vol[pk]
                            if not isinstance(pg, h5py.Group):
                                continue
                            if 'delta_py' in pg:
                                return 'pixel'
                            if 'delta_wire' in pg:
                                return 'wire'
                    elif 'delta_py' in vol:
                        return 'pixel'
                    elif 'delta_wire' in vol:
                        return 'wire'
                break
        return None

    def _detect_num_time_steps(self):
        """Read the per-file tick count from ``/config`` or first-event attrs.

        Returns ``None`` if no file carries the attr (older fixtures); a
        downstream ``Densify`` then needs ``num_time_steps`` supplied explicitly.
        """
        for path in self.h5_files:
            try:
                with h5py.File(path, 'r', libver='latest', swmr=True) as f:
                    if 'config' in f and 'num_time_steps' in f['config'].attrs:
                        return int(f['config'].attrs['num_time_steps'])
                    for ek in f:
                        if ek.startswith('event_'):
                            if 'num_time_steps' in f[ek].attrs:
                                return int(f[ek].attrs['num_time_steps'])
                            break
            except Exception as e:
                log.warning("num_time_steps detection failed on %s: %s", path, e)
                continue
        return None

    def _plane_has_payload(self, g):
        """True if this group contains a plane's sparse data payload."""
        if self.readout_type == 'pixel':
            return 'delta_py' in g
        return 'delta_wire' in g

    def _iter_planes(self, evt):
        """Yield (plane_label, h5py.Group) for each plane in an event.

        Handles both formats:
          - Old: planes directly under event (east_U, east_V, ...)
          - New: planes under volume_N/ subgroups
        """
        for key in evt:
            obj = evt[key]
            if not isinstance(obj, h5py.Group):
                continue
            if key.startswith('volume_'):
                vol_label = key
                for plane_key in obj:
                    pg = obj[plane_key]
                    if isinstance(pg, h5py.Group) and self._plane_has_payload(pg):
                        yield f'{vol_label}_{plane_key}', pg
            elif self._plane_has_payload(obj):
                yield key, obj

    def _decode_plane_wire(self, g):
        """Decode one wire plane's delta-encoded sparse data."""
        wire_start = int(g.attrs['wire_start'])
        time_start = int(g.attrs['time_start'])

        # cumsum directly into int32 + in-place add avoids the wide-int64
        # accumulator and the extra full-size copies of the old expression.
        wire = np.cumsum(g['delta_wire'][:], dtype=np.int32); wire += wire_start
        time = np.cumsum(g['delta_time'][:], dtype=np.int32); time += time_start

        values = self._decode_values(g)
        return wire, time, values

    def _decode_plane_pixel(self, g):
        """Decode one pixel plane's delta-encoded sparse data."""
        py_start = int(g.attrs['py_start'])
        pz_start = int(g.attrs['pz_start'])
        time_start = int(g.attrs['time_start'])

        py = np.cumsum(g['delta_py'][:], dtype=np.int32); py += py_start
        pz = np.cumsum(g['delta_pz'][:], dtype=np.int32); pz += pz_start
        time = np.cumsum(g['delta_time'][:], dtype=np.int32); time += time_start

        values = self._decode_values(g)
        return py, pz, time, values

    def _decode_values(self, g):
        """Shared value decoding (handles uint16 digitization)."""
        raw_values = g['values'][:]
        vals = raw_values.astype(np.float32)
        if self.decode_digitization and raw_values.dtype == np.uint16:
            vals -= int(g.attrs.get('pedestal', 0))  # in-place, no extra copy
        return vals

    def read_event(self, idx):
        """Read one event, return dict with plane-namespaced sparse arrays.

        Wire returns keys like:
            sensor.{plane}.{wire, time, value}
        Pixel returns keys like:
            sensor.{plane}.{py, pz, time, value}
        """
        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data_dict = {}
        for plane_label, pg in self._iter_planes(evt):
            if self.planes != 'all' and plane_label not in self.planes:
                continue

            prefix = f'sensor.{plane_label}'
            if self.readout_type == 'pixel':
                py, pz, time, values = self._decode_plane_pixel(pg)
                data_dict[f'{prefix}.py'] = py
                data_dict[f'{prefix}.pz'] = pz
                data_dict[f'{prefix}.time'] = time
                data_dict[f'{prefix}.value'] = values
            else:
                wire, time, values = self._decode_plane_wire(pg)
                data_dict[f'{prefix}.wire'] = wire
                data_dict[f'{prefix}.time'] = time
                data_dict[f'{prefix}.value'] = values
                # Surface the fixed wire count for densify + the pedestal for
                # digitize (attrs win over the ctor fallbacks). A missing entry
                # is tolerated — the dataset/transform report it.
                if 'n_wires' in pg.attrs:
                    self.n_wires_per_plane[plane_label] = int(pg.attrs['n_wires'])
                if 'pedestal' in pg.attrs:
                    self.pedestal_per_plane[plane_label] = int(pg.attrs['pedestal'])

        return data_dict
