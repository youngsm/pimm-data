import os
#!/usr/bin/env python3
"""Micro-benchmark: current vs optimized hits decode+merge.

The reader currently, per plane, does
    wires = np.repeat(center, gs).astype(int32) + delta.astype(int32)
then jaxtpc._merge_plane_dotted does np.stack([w,t],1).astype(f32) per
plane and concatenate across planes — several full-size intermediate
copies. The optimized path preallocates the final (N,2) float32 coord and
(N,1) energy and fills per-plane slices directly (no int32 intermediates,
no stack, no concatenate).

Run after the loader profiles (it is CPU-timing sensitive).
"""
import time
import numpy as np
import h5py

try:  # readable against blosc/zstd/lz4 output too
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Was a hardcoded path under one developer's home that no longer exists.
SRC = os.environ.get('BENCH_SHARD', '')
if not SRC:
    raise SystemExit('set BENCH_SHARD=<a hits shard .h5> to run this benchmark')


def load_planes(ev='event_136'):
    f = h5py.File(SRC, 'r')
    planes = []
    for vk in f[ev]:
        if not vk.startswith('volume_'):
            continue
        for pk in f[ev][vk]:
            g = f[ev][vk][pk]
            if isinstance(g, h5py.Group) and 'delta_wires' in g:
                planes.append({k: g[k][:] for k in (
                    'group_ids', 'group_sizes', 'center_wires', 'center_times',
                    'peak_charges', 'delta_wires', 'delta_times', 'charges_u16')})
    f.close()
    return planes


def current(planes):
    """Mirror reader _decode_plane_wire + jaxtpc _merge_plane_dotted."""
    dec = []
    for r in planes:
        gs = r['group_sizes'].astype(np.int32)
        w = np.repeat(r['center_wires'], gs).astype(np.int32) + r['delta_wires'].astype(np.int32)
        t = np.repeat(r['center_times'], gs).astype(np.int32) + r['delta_times'].astype(np.int32)
        ch = np.repeat(r['peak_charges'], gs) * r['charges_u16'].astype(np.float32) / 65535.0
        dec.append((w, t, ch))
    coord = np.concatenate([np.stack([w, t], axis=1).astype(np.float32) for w, t, _ in dec], axis=0)
    energy = np.concatenate([ch[:, None].astype(np.float32) for _, _, ch in dec], axis=0)
    return coord, energy


def optimized(planes):
    """Preallocate final arrays; fill per-plane slices directly as float32."""
    ns = [len(r['delta_wires']) for r in planes]
    N = sum(ns)
    coord = np.empty((N, 2), np.float32)
    energy = np.empty((N, 1), np.float32)
    inv = np.float32(1.0 / 65535.0)
    off = 0
    for r, n in zip(planes, ns):
        gs = r['group_sizes']
        sl = slice(off, off + n)
        # column 0: center_wire (repeated) + delta_wire, written as float32
        coord[sl, 0] = np.repeat(r['center_wires'], gs)
        coord[sl, 0] += r['delta_wires']
        coord[sl, 1] = np.repeat(r['center_times'], gs)
        coord[sl, 1] += r['delta_times']
        energy[sl, 0] = np.repeat(r['peak_charges'], gs)
        energy[sl, 0] *= r['charges_u16'] * inv
        off += n
    return coord, energy


def optimized2(planes):
    """Concatenate the small per-group arrays, then 3 big repeats across all
    planes at once (no per-plane Python loop, no charge temp)."""
    gs = np.concatenate([r['group_sizes'] for r in planes]).astype(np.intp)
    cw = np.concatenate([r['center_wires'] for r in planes])
    ct = np.concatenate([r['center_times'] for r in planes])
    pc = np.concatenate([r['peak_charges'] for r in planes])
    dw = np.concatenate([r['delta_wires'] for r in planes])
    dt = np.concatenate([r['delta_times'] for r in planes])
    u16 = np.concatenate([r['charges_u16'] for r in planes])
    N = dw.shape[0]
    coord = np.empty((N, 2), np.float32)
    energy = np.empty((N, 1), np.float32)
    coord[:, 0] = np.repeat(cw, gs); coord[:, 0] += dw
    coord[:, 1] = np.repeat(ct, gs); coord[:, 1] += dt
    energy[:, 0] = np.repeat(pc, gs); energy[:, 0] *= u16; energy[:, 0] *= np.float32(1.0 / 65535.0)
    return coord, energy


def optimized3(planes):
    """3 single big repeats over concatenated *group-level* arrays (small),
    then add the per-entry deltas per-plane into slices (no big concat)."""
    ns = [len(r['delta_wires']) for r in planes]
    N = sum(ns)
    gs = np.concatenate([r['group_sizes'] for r in planes]).astype(np.intp)
    cw = np.concatenate([r['center_wires'] for r in planes])
    ct = np.concatenate([r['center_times'] for r in planes])
    pc = np.concatenate([r['peak_charges'] for r in planes])
    coord = np.empty((N, 2), np.float32)
    energy = np.empty((N, 1), np.float32)
    coord[:, 0] = np.repeat(cw, gs)
    coord[:, 1] = np.repeat(ct, gs)
    energy[:, 0] = np.repeat(pc, gs)
    off = 0
    for r, n in zip(planes, ns):
        sl = slice(off, off + n)
        coord[sl, 0] += r['delta_wires']
        coord[sl, 1] += r['delta_times']
        energy[sl, 0] *= r['charges_u16']
        off += n
    energy[:, 0] *= np.float32(1.0 / 65535.0)
    return coord, energy


def med(fn, planes, reps=10):
    fn(planes); fn(planes)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(planes); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1000


if __name__ == '__main__':
    for ev in ('event_136', 'event_086', 'event_000'):
        planes = load_planes(ev)
        n = sum(len(r['delta_wires']) for r in planes)
        c0, e0 = current(planes)
        c1, e1 = optimized(planes)
        c3, e3 = optimized3(planes)
        ok1 = np.allclose(c0, c1) and np.allclose(e0, e1)
        ok3 = np.allclose(c0, c3) and np.allclose(e0, e3)
        tc = med(current, planes); to = med(optimized, planes); to3 = med(optimized3, planes)
        print(f'{ev}: {n:,} ent | current={tc:6.1f}  opt1(per-plane)={to:6.1f} '
              f'({tc/to:.2f}x, {ok1})  opt3(big-repeat)={to3:6.1f} ({tc/to3:.2f}x, {ok3})')
