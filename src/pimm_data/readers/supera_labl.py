"""
SuperaLablReader — JAXTPC label tables read directly from pysupera output.

Drop-in replacement for :class:`JAXTPCLablReader` when a production has no
``labl/`` shards but does have Supera files. It returns the same flat
``labl_v{N}_{col}`` dict, so the dataset's existing decoration
(``deposit_to_track`` for step, ``group_to_track`` for hits) is unchanged.

How the join works
------------------
Supera assigns every JAXTPC hits group to a fragment
(``groups/fragment_id`` -> ``particles.id``). Each Geant4 track maps to at
most one fragment, so the label table is keyed by the Geant4 track id that
the hits file's ``group_to_track`` already holds, and each track takes the
labels of its fragment. Tracks that never deposit charge on the sensor have
no fragment; they are left out of the table and decorate as -1.

Output columns per volume (``labl_v{N}_*``)::

    track_ids         (T,)   Geant4 track id (primary key)
    track_motif       (T,)   PILArNet 5-class motif, see SEM_TYPE_TO_MOTIF
    track_sem         (T,)   raw Supera fragment semantic type
    track_pdg         (T,)   PDG of the fragment's root particle
    track_cluster     (T,)   Supera fragment id
    track_instance    (T,)   Supera particle instance id
    track_interaction (T,)   Supera interaction id
    track_ancestor    (T,)   Supera ancestor id
    deposit_to_track  (N_v,) per-step-deposit Geant4 track id
"""

import glob
import os
import re

import h5py
import numpy as np

# labl column suffix -> Supera ``particles/`` dataset it is copied from.
PARTICLE_COLUMNS = {
    'sem': 'frag_sem_type',
    'pdg': 'pdg',
    'cluster': 'frag_id',
    'instance': 'inst_id',
    'interaction': 'interaction_id',
    'ancestor': 'ancestor_id',
}

# Supera fragment semantic type -> PILArNet motif class
# (0 shower, 1 track, 2 michel, 3 delta, 4 led). Inferred from the PDG and
# parent content of each type in the doraemon test campaign; confirm against
# the pysupera enum. Types not listed (e.g. 6) map to -1 (ignored).
SEM_TYPE_TO_MOTIF = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4}
LED_MOTIF = 4  # groups Supera flags as low-energy (is_le) are always 'led'

_JOB_ID = re.compile(r'_j(\d+)-\d+\.h5$')


class SuperaLablReader:
    """Reads per-volume track label tables from pysupera files.

    Parameters
    ----------
    supera_root : str
        Campaign-level Supera directory (e.g. ``.../data/supera_pixel``),
        laid out as ``<supera_root>/<group>/<chunk>/<name>_j<job>-<job>.h5``.
    """

    def __init__(self, supera_root):
        self.supera_root = supera_root
        self._supera_files = None  # {job id: path}, built on first use
        self._handles = {}
        self._pid = None

    def read_event(self, jaxtpc_file, event_key):
        """Label tables for ``event_key`` of the JAXTPC shard ``jaxtpc_file``.

        ``jaxtpc_file`` may be the step, sensor or hits shard of the event;
        the hits shard (for ``group_to_track`` / ``deposit_to_group``) and the
        Supera file are located from it.
        """
        jaxtpc = self._open(jaxtpc_file)
        supera = self._open(self._supera_file_for(jaxtpc))
        hits_event = self._open(_sibling_shard(jaxtpc_file, 'hits'))[event_key]

        row = _supera_event_row(supera, jaxtpc[event_key].attrs['event_id'])
        particles = _event_particles(supera, row)

        labl = {}
        for volume_key in hits_event:
            if not volume_key.startswith('volume_'):
                continue
            volume = int(volume_key.split('_')[1])
            tables = _volume_tables(supera, row, volume,
                                    hits_event[volume_key], particles)
            for column, values in tables.items():
                labl[f'labl_v{volume}_{column}'] = values
        return labl

    # -- file handling ------------------------------------------------------

    def _open(self, path):
        """Cached read-only handle; dropped (not closed) after a fork, since
        HDF5 handles must not be shared across processes."""
        if self._pid != os.getpid():
            self._handles = {}
            self._pid = os.getpid()
        if path not in self._handles:
            self._handles[path] = h5py.File(path, 'r', libver='latest', swmr=True)
        return self._handles[path]

    def _supera_file_for(self, jaxtpc):
        """Supera file produced from the same edep-sim job as ``jaxtpc``."""
        if self._supera_files is None:
            pattern = os.path.join(self.supera_root, '*', '*', '*.h5')
            self._supera_files = {int(_JOB_ID.search(p).group(1)): p
                                  for p in glob.glob(pattern) if _JOB_ID.search(p)}
        source_file = str(jaxtpc['config'].attrs['source_file'])
        return self._supera_files[int(_JOB_ID.search(source_file).group(1))]


def _sibling_shard(path, modality):
    """``<chunk>/<mod>/<run>/<name>_<mod>_<tag>.h5`` -> same shard of ``modality``."""
    run_dir, filename = os.path.split(path)
    modality_dir, run = os.path.split(run_dir)
    chunk_dir, source_modality = os.path.split(modality_dir)
    filename = filename.replace(f'_{source_modality}_', f'_{modality}_')
    return os.path.join(chunk_dir, modality, run, filename)


def _supera_event_row(supera, event_id):
    """Row of ``event_id`` in the Supera file's per-event tables."""
    rows = np.flatnonzero(supera['events/event_id'][:] == event_id)
    if len(rows) != 1:
        raise KeyError(f"event_id {event_id} found {len(rows)} times in {supera.filename}")
    return int(rows[0])


def _event_particles(supera, row):
    """This event's slice of the Supera particle table."""
    offsets = supera['events/offsets']
    rows = slice(int(offsets[row]), int(offsets[row + 1]))
    columns = {'id', *PARTICLE_COLUMNS.values()}
    return {c: supera['particles'][c][rows] for c in columns}


def _volume_group_rows(supera, row, volume, n_groups):
    """Supera's per-group rows for one event and volume.

    ``groups/*`` has one row per JAXTPC hits group: per event, all of volume
    0's groups, then volume 1's, ...
    """
    event_start = supera['groups/offsets'][row]
    first_volume = supera['groups/volume_offsets_index'][row]
    volume_start = supera['groups/volume_offsets'][first_volume + volume]
    start = int(event_start + volume_start)
    return slice(start, start + n_groups)


def _volume_tables(supera, row, volume, hits_volume, particles):
    """Track label tables for one volume of one event."""
    group_to_track = hits_volume['group_to_track'][:]
    deposit_to_group = hits_volume['deposit_to_group'][:]

    groups = _volume_group_rows(supera, row, volume, len(group_to_track))
    group_fragment = supera['groups/fragment_id'][groups]
    group_is_le = supera['groups/is_le'][groups].astype(bool)

    # Every group of a track shares its fragment and is_le flag, so the first
    # labelled group of each track speaks for the track.
    labelled = group_fragment >= 0
    track_ids, first_group = np.unique(group_to_track[labelled], return_index=True)
    track_fragment = group_fragment[labelled][first_group]
    track_is_le = group_is_le[labelled][first_group]

    by_id = np.argsort(particles['id'])
    particle_row = by_id[np.searchsorted(particles['id'], track_fragment, sorter=by_id)]

    tables = {'track_ids': track_ids.astype(np.int32)}
    for suffix, column in PARTICLE_COLUMNS.items():
        tables[f'track_{suffix}'] = particles[column][particle_row].astype(np.int32)

    motif = np.full(len(track_ids), -1, dtype=np.int32)
    for sem_type, motif_class in SEM_TYPE_TO_MOTIF.items():
        motif[tables['track_sem'] == sem_type] = motif_class
    motif[track_is_le] = LED_MOTIF
    tables['track_motif'] = motif
    tables['deposit_to_track'] = group_to_track[deposit_to_group].astype(np.int32)
    return tables
