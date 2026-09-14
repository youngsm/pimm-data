"""Shared dataset-spec helpers for the loader profiling scripts.

Lets ``benchmark_loader.py``, ``profile_loader.py`` and
``profile_scaling.py`` target either the LUCiD (``wc_*``) or JAXTPC
(``sim_*``) v3 datasets through a single ``--dataset {lucid,jaxtpc}``
flag, so the three scripts share one code path per dataset instead of
hard-coding LUCiD.

What differs between the two datasets:

* **file prefix / dataset_name** — ``wc`` vs ``sim``.
* **layout** — LUCiD shards are flat under each modality dir
  (``step/wc_step_*.h5``); JAXTPC nests one level deeper by run
  (``step/run_00266285XX/sim_step_*.h5``), so a run maps to ``split``.
* **modalities** — LUCiD has ``labl``; the doraemon JAXTPC sample does
  not, so JAXTPC defaults to ``(step, sensor, hits)``.
* **stream schema** — the minimal "tensorize one stream" transform pulls
  different keys (LUCiD ``hits`` carries a bare ``time``; JAXTPC folds
  time into the 2-D wire ``coord`` and adds ``instance``).
* **raw per-event read** — LUCiD events are flat column datasets; JAXTPC
  events are nested ``volume_N[/plane]`` groups, so the raw-I/O baseline
  reads every dataset under the event group instead of a fixed list.

The raw-read workers are module-level (not closures) so
``multiprocessing.Pool`` can pickle them under the ``fork`` start method.
"""
import glob
import os

import h5py
import numpy as np

try:  # register blosc/zstd/lz4 HDF5 filters so those files are readable
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


# --- default dataset locations -------------------------------------------
LUCID_DEFAULT_ROOT = '/sdf/data/neutrino/cjesus/DORAEMON/OLD/SK_like/config_000013'
# Was a hardcoded path under one developer's home that no longer exists.
JAXTPC_DEFAULT_ROOT = os.environ.get(
    'JAXTPC_DATA_ROOT', '/sdf/data/neutrino/doraemon')
JAXTPC_DEFAULT_SPLIT = 'run_0026628550'   # one complete run: 100 shards, 20k events

DATASETS = ('lucid', 'jaxtpc')


def default_root(dataset):
    return JAXTPC_DEFAULT_ROOT if dataset == 'jaxtpc' else LUCID_DEFAULT_ROOT


def default_split(dataset):
    """Subdirectory under each modality dir. JAXTPC keys this by run."""
    return JAXTPC_DEFAULT_SPLIT if dataset == 'jaxtpc' else ''


def default_modalities(dataset):
    if dataset == 'jaxtpc':
        return ('step', 'sensor', 'hits')        # doraemon has no labl/
    return ('step', 'sensor', 'hits', 'labl')


def file_prefix(dataset):
    return 'sim' if dataset == 'jaxtpc' else 'wc'


def default_transform(dataset):
    """Minimal "extract one stream and tensorize" transform.

    This is the bare requirement for tensor-batched training; heavier
    augmentations are model-dependent and would conflate loader
    throughput with transform throughput. Both datasets collect the
    heaviest stream (``hits``) so the numbers are comparable.
    """
    if dataset == 'jaxtpc':
        # JAXTPC wire hits: coord is (wire, time); time is not a separate
        # key, and each entry carries a group instance id.
        return [dict(type='Collect', stream='hits',
                     keys=['coord', 'energy', 'instance'],
                     feat_keys=['energy'])]
    return [dict(type='Collect', stream='hits',
                 keys=['coord', 'energy', 'time'],
                 feat_keys=['energy', 'time'])]


def transform_variant(dataset, name):
    """Transform stacks that isolate the Collect / ToTensor ordering cost.

    All three collect the same heaviest stream (``hits``) from the nested
    ``{'step': {...}, 'sensor': {...}, 'hits': {...}}`` dict; they differ
    only in whether a global ``ToTensor`` runs, and on which side of
    ``Collect`` it sits:

      * ``loading_only``  — ``[Collect(hits)]``. The prior baseline.
        ``Collect`` already tensorizes the keys it keeps and drops the
        other streams, so this is the cheapest path.
      * ``collect_first`` — ``[Collect(hits), ToTensor]``. ``Collect`` drops
        ``step`` + ``sensor`` first, so the trailing global ``ToTensor``
        sees only the (already-tensorized) kept keys → every node is a
        no-op. This is the fix for the training config.
      * ``totensor_first`` — ``[ToTensor, Collect(hits)]``. The order the
        ``jaxtpc_seg`` training config uses: the global ``ToTensor``
        recurses the whole nested dict and tensorizes the ``step`` +
        ``sensor`` streams that ``Collect`` then discards. That recursion
        over the discarded streams is the wasted work the reorder removes.
    """
    base = default_transform(dataset)            # [Collect(...)]
    if name in ('loading_only', 'default'):
        return base
    if name == 'collect_first':
        return base + [dict(type='ToTensor')]
    if name == 'totensor_first':
        return [dict(type='ToTensor')] + base
    raise ValueError(f'unknown transform variant: {name!r} '
                     "(loading_only | collect_first | totensor_first)")


# --- dataset construction (imports torch/pimm_data lazily) ---------------
def build_dataset(dataset, data_root, split=None, modalities=None,
                  transform='default'):
    """Construct a LUCiDDataset or JAXTPCDataset with sensible defaults."""
    if transform == 'default':
        transform = default_transform(dataset)
    if modalities is None:
        modalities = default_modalities(dataset)
    if split is None:
        split = default_split(dataset)

    if dataset == 'jaxtpc':
        from pimm_data import JAXTPCDataset
        return JAXTPCDataset(
            data_root=data_root, split=split, dataset_name='sim',
            modalities=tuple(modalities), transform=transform)
    from pimm_data import LUCiDDataset
    return LUCiDDataset(
        data_root=data_root, split=split, dataset_name='wc',
        modalities=tuple(modalities), transform=transform)


# --- raw-h5py file discovery ---------------------------------------------
def modality_files(dataset, data_root, modality, split=None):
    """Sorted shard list for one modality, honoring the split subdir.

    Tries ``<root>/<modality>/<split>/<prefix>_<modality>_*.h5`` first,
    then falls back to the flat ``<root>/<modality>/<prefix>_...`` layout.
    """
    if split is None:
        split = default_split(dataset)
    prefix = file_prefix(dataset)
    bases = []
    if split:
        bases.append(os.path.join(data_root, modality, split))
    bases.append(os.path.join(data_root, modality))
    for base in bases:
        files = sorted(glob.glob(os.path.join(base, f'{prefix}_{modality}_*.h5')))
        if files:
            return files
    return []


# --- per-event raw reads (mirror what the readers touch) -----------------
# LUCiD: flat per-event column datasets. JAXTPC: nested volume_N[/plane]
# groups, read generically (the readers consume essentially every dataset
# under the event group, so reading them all is a faithful I/O ceiling).

_LUCID_STEP_KEYS = (
    'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
    'dir_x', 'dir_y', 'dir_z', 'edep', 'time', 'track_idx',
    'beta_start', 'n_cherenkov', 'contained',
)


def _read_event_group_all(evt):
    """Recursively read every dataset under an event group; return nbytes."""
    total = {'n': 0}

    def _visit(_name, obj):
        if isinstance(obj, h5py.Dataset):
            arr = obj[()] if obj.shape == () else obj[:]
            total['n'] += getattr(arr, 'nbytes', 0)

    evt.visititems(_visit)
    return total['n']


def read_event_step(dataset, f, event_key):
    """Read one event's step arrays from an open step file. Returns nbytes."""
    if event_key not in f:
        return 0
    evt = f[event_key]
    if dataset == 'jaxtpc':
        return _read_event_group_all(evt)
    nbytes = 0
    for key in _LUCID_STEP_KEYS:
        if key in evt:
            arr = evt[key][:]
            nbytes += arr.nbytes
    return nbytes


def n_events_in(path):
    with h5py.File(path, 'r', libver='latest', swmr=True) as f:
        return int(f['config'].attrs['n_events'])


def per_event_step_read(args):
    """Picklable Pool/Thread worker: read ``n`` step events from one shard.

    ``args = (dataset, file_path, start, n)``. Returns
    ``(elapsed_s, n_events, bytes_read)``. Used by profile_scaling.py.
    """
    import time
    dataset, file_path, start, n = args
    bytes_read = 0
    with h5py.File(file_path, 'r', libver='latest', swmr=True) as f:
        n_in_file = int(f['config'].attrs['n_events'])
        t0 = time.perf_counter()
        for i in range(n):
            ek = f'event_{(start + i) % n_in_file:03d}'
            bytes_read += read_event_step(dataset, f, ek)
        elapsed = time.perf_counter() - t0
    return elapsed, n, bytes_read


# --- per-modality reader construction (profile_loader.py) ----------------
def build_readers(dataset, data_root, split=None, modalities=None):
    """Build + open one reader per modality. Returns list of (name, reader).

    Readers share the (data_root, split, dataset_name) signature across
    both datasets, so construction differs only in the class and the
    per-modality root.
    """
    if split is None:
        split = default_split(dataset)
    if modalities is None:
        modalities = default_modalities(dataset)
    name = file_prefix(dataset)

    if dataset == 'jaxtpc':
        from pimm_data.readers.jaxtpc_step import JAXTPCStepReader
        from pimm_data.readers.jaxtpc_sensor import JAXTPCSensorReader
        from pimm_data.readers.jaxtpc_hits import JAXTPCHitsReader
        classes = {'step': JAXTPCStepReader, 'sensor': JAXTPCSensorReader,
                   'hits': JAXTPCHitsReader}
    else:
        from pimm_data.readers.lucid_step import LUCiDStepReader
        from pimm_data.readers.lucid_sensor import LUCiDSensorReader
        from pimm_data.readers.lucid_hits import LUCiDHitsReader
        from pimm_data.readers.lucid_labl import LUCiDLablReader
        classes = {'step': LUCiDStepReader, 'sensor': LUCiDSensorReader,
                   'hits': LUCiDHitsReader, 'labl': LUCiDLablReader}

    readers = []
    for m in modalities:
        if m not in classes:
            continue
        r = classes[m](data_root=os.path.join(data_root, m), split=split,
                       dataset_name=name)
        r.h5py_worker_init()
        readers.append((m, r))
    return readers


# --- multi-modality raw-h5py baseline (profile_loader.py) ----------------
_LUCID_SENSOR_KEYS = ('sensor_idx', 'PE', 'T')
_LUCID_HITS_KEYS = ('sensor_idx', 'particle_idx', 'PE', 'T')


def _read_lucid_labl_event(evt):
    """Mirror LUCiDLablReader's per-event reads. Returns nbytes."""
    nbytes = 0
    if 'per_event' in evt:
        pe = evt['per_event']
        for k in ('t0', 'contained'):
            if k in pe:
                nbytes += np.asarray(pe[k][()]).nbytes
    if 'per_particle' in evt:
        pp = evt['per_particle']
        for k in ('category', 'contained', 'genealogy_data',
                  'genealogy_offsets', 'ext_genealogy_data',
                  'ext_genealogy_offsets'):
            if k in pp:
                nbytes += pp[k][:].nbytes
    if 'per_track' in evt:
        pt = evt['per_track']
        for k in ('track_id', 'pdg', 'parent_id', 'particle_idx',
                  'ancestor', 'interaction', 'initial_energy', 'n_cherenkov'):
            if k in pt:
                nbytes += pt[k][:].nbytes
    return nbytes


def _read_event_modality(dataset, f, event_key, modality):
    """nbytes read for one event of one modality from an open file."""
    if event_key not in f:
        return 0
    evt = f[event_key]
    if dataset == 'jaxtpc':
        return _read_event_group_all(evt)
    # LUCiD: curated per-modality columns (verbatim from the original
    # profile_loader baseline, so documented numbers are reproduced).
    if modality == 'step':
        return read_event_step(dataset, f, event_key)
    keys = {'sensor': _LUCID_SENSOR_KEYS, 'hits': _LUCID_HITS_KEYS}.get(modality)
    if keys is not None:
        return sum(evt[k][:].nbytes for k in keys if k in evt)
    if modality == 'labl':
        return _read_lucid_labl_event(evt)
    return 0


def raw_reader(dataset, data_root, split=None, modalities=None):
    """Build a raw-h5py per-event reader over all loaded modalities.

    Establishes the pure-I/O ceiling: opens every shard once, indexes by
    the step (canonical) modality's per-shard event counts, and on each
    ``read(idx)`` touches the same datasets the readers consume.

    Returns ``(read_fn, close_fn, n_total_events)``.
    """
    if split is None:
        split = default_split(dataset)
    if modalities is None:
        modalities = default_modalities(dataset)
    # step is the canonical index source; fall back to the first modality.
    index_mod = 'step' if 'step' in modalities else modalities[0]

    files = {m: modality_files(dataset, data_root, m, split=split)
             for m in modalities}
    assert files[index_mod], (
        f'no {index_mod} shards under {data_root} (split={split!r})')

    counts = [n_events_in(p) for p in files[index_mod]]
    cumlens = np.cumsum(counts)
    handles = {m: [h5py.File(p, 'r', libver='latest', swmr=True)
                   for p in files[m]] for m in modalities}

    def locate(idx):
        i = int(np.searchsorted(cumlens, idx, side='right'))
        local = idx - (int(cumlens[i - 1]) if i > 0 else 0)
        return i, f'event_{local:03d}'

    def read(idx):
        fi, ek = locate(idx)
        for m in modalities:
            if fi < len(handles[m]):
                _read_event_modality(dataset, handles[m][fi], ek, m)

    def close():
        for hs in handles.values():
            for fh in hs:
                fh.close()

    return read, close, int(cumlens[-1])
