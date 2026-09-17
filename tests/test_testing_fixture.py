

def test_step_shards_carry_the_geometry_real_ones_do(tmp_path):
    """`config/{num_wires, volume_ranges}`, because a consumer reads them.

    Real JAXTPC step shards have both; this fixture had neither, so anything
    reading them could not be exercised without production data.
    helix/scripts/dump_probe_truth.py:240 reads config/volume_ranges and died
    with KeyError on a fixture-built corpus -- which made the fixture unusable
    as a full-pipeline smoke input, exactly where it matters most: a handover
    target has the code and no data.
    """
    import h5py
    import numpy as np

    from pimm_data.testing import make_jaxtpc_sample

    root = make_jaxtpc_sample(str(tmp_path / "d"), dataset_name="sim_wire",
                              n_events=2, n_files=2, n_volumes=2,
                              readout_type="wire")
    for i in range(2):
        with h5py.File(f"{root}/step/sim_wire_step_{i:04d}.h5") as f:
            cfg = f["config"]
            assert "num_wires" in cfg and "volume_ranges" in cfg
            nw, vr = cfg["num_wires"][:], cfg["volume_ranges"][:]
            assert nw.shape == (2, 3) and nw.dtype == np.int32
            assert vr.shape == (2, 3, 2) and vr.dtype == np.float32
            # volumes tile x and share y/z, as in a real shard
            assert vr[0, 0, 1] == vr[1, 0, 0]
            assert (vr[0, 1] == vr[1, 1]).all()
            # every range is (min, max), not reversed
            assert (vr[..., 0] < vr[..., 1]).all()


def test_pixel_readout_gets_one_plane_of_geometry(tmp_path):
    import h5py

    from pimm_data.testing import make_jaxtpc_sample

    root = make_jaxtpc_sample(str(tmp_path / "p"), dataset_name="sim_pix",
                              n_events=2, n_files=1, n_volumes=2,
                              readout_type="pixel")
    with h5py.File(f"{root}/step/sim_pix_step_0000.h5") as f:
        assert f["config"]["num_wires"][:].shape == (2, 1)
        assert f["config"]["volume_ranges"][:].shape == (2, 3, 2)
