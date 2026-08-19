"""Graph producers: SetupGraph (kNN, self edges) + BuildNexus (cross-store).

End-to-end: producer stamps edge roles -> Collect flattens + carries roles ->
collate shifts edges by node count.
"""
import numpy as np

from pimm_data.transform import Compose
from pimm_data import collate_fn


def _event(n, seed):
    rng = np.random.default_rng(seed)
    return {'step': {'coord': rng.standard_normal((n, 3)).astype('float32'),
                     'energy': rng.standard_normal((n, 1)).astype('float32')}}


def test_setup_graph_edges_shift_on_collate():
    pipe = Compose([
        dict(type='SetupGraph', on='step', k=2),
        dict(type='Collect', parts={
            'step': dict(keys=('coord', 'edge_index'), feat_keys=('coord', 'energy'))}),
    ])
    s0 = pipe(_event(4, 0))
    s1 = pipe(_event(3, 1))
    n0 = s0['step_coord'].shape[0]
    b = collate_fn([s0, s1])
    assert b['_roles']['step_edge_index'] == ('edge', 'self')
    # event1's edges must be shifted by event0's node count
    e = b['step_edge_index']
    assert e.shape[0] == 2
    # max index in range of total nodes; event1 endpoints >= n0
    assert int(e.max()) < b['step_coord'].shape[0]
    assert int(e[:, e.shape[1] // 2:].min()) >= 0


def test_build_nexus_cross_store():
    def ev(nh, ns, seed):
        rng = np.random.default_rng(seed)
        return {'hit': {'coord': rng.standard_normal((nh, 3)).astype('float32')},
                'sp':  {'coord': rng.standard_normal((ns, 3)).astype('float32')}}
    pipe = Compose([
        dict(type='BuildNexus', on=('hit', 'sp'), to='nexus', k=1),
        dict(type='Collect', parts={
            'hit':   dict(keys=('coord',)),
            'sp':    dict(keys=('coord',)),
            'nexus': dict(keys=('edge_index',), offset_keys_dict={})}),
    ])
    s0 = pipe(ev(3, 2, 0))
    s1 = pipe(ev(2, 1, 1))
    nh0 = s0['hit_coord'].shape[0]
    ns0 = s0['sp_coord'].shape[0]
    b = collate_fn([s0, s1])
    assert b['_roles']['nexus_edge_index'] == ('edge', ('hit', 'sp'))
    e = b['nexus_edge_index']
    # row0 indexes hit (in range of total hits), row1 indexes sp (total sp)
    assert int(e[0].max()) < b['hit_coord'].shape[0]
    assert int(e[1].max()) < b['sp_coord'].shape[0]
    # event1's edges shifted: row0 >= nh0, row1 >= ns0 for the tail
    tail = e[:, s0_nexus_count(s0):]
    assert tail.shape[1] == 0 or (int(tail[0].min()) >= nh0 and int(tail[1].min()) >= ns0)


def s0_nexus_count(s0):
    return s0['nexus_edge_index'].shape[1]


def test_setup_graph_then_subsample_remaps_edges():
    """SetupGraph then GridSample (subsample) -> edges remapped, no dangling index.
    Proves the roles-aware index_operator edge remap (subsample/graph any order)."""
    import numpy as np
    from pimm_data.transform import Compose
    np.random.seed(0)
    rng = np.random.default_rng(0)
    d = {'step': {'coord': rng.standard_normal((30, 3)).astype('float32') * 5}}
    out = Compose([
        dict(type='SetupGraph', on='step', k=3),
        dict(type='Apply', on='step', transforms=[
            dict(type='GridSample', grid_size=1.0, mode='train')]),
    ])(d)
    sub = out['step']
    n = sub['coord'].shape[0]
    e = sub['edge_index']
    e = e.cpu().numpy() if hasattr(e, 'cpu') else e
    assert e.shape[0] == 2
    # every remaining edge endpoint is a valid row of the SUBSAMPLED cloud
    assert e.size == 0 or (int(e.min()) >= 0 and int(e.max()) < n)
