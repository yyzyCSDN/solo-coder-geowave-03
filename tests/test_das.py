import numpy as np
import pytest

from geowave.das import (
    FiberPath, ChannelGeometry, DASGather, load_strain_rate,
    gauge_average_operator, gauge_difference_operator, reconstruct_channels,
    directional_sensitivity_p, directional_sensitivity_s, axial_field,
    QualityFlag, assess_quality, break_segments, detect_reversed,
    resample_spatial, resample_time, lineage_from_operator,
    DASConfig, process_das, RowStatus,
    read_fiber_csv, read_channel_csv, read_strain_rate_npy,
)


@pytest.fixture
def curved_fiber():
    # east for 40 m, half-circle U-turn (radius 3 m, arc length 3*pi), west for 40 m
    theta = np.linspace(-np.pi / 2, np.pi / 2, 2000)
    ux = 40.0 + 3.0 * np.cos(theta)
    uy = 3.0 + 3.0 * np.sin(theta)
    x = np.concatenate([np.linspace(0, 40, 401), ux[1:],
                        np.linspace(40, 0, 401)[1:]])
    y = np.concatenate([np.zeros(401), uy[1:],
                        np.full(400, 6.0)])
    return FiberPath.from_vertices(x, y)


@pytest.fixture
def regular_fiber():
    return FiberPath.from_vertices(np.linspace(0, 100, 101), np.zeros(101))


def make_gather(geom, n_t=200, dt=0.002, seed=0):
    rng = np.random.default_rng(seed)
    return DASGather(rng.standard_normal((geom.n_channels, n_t)), geom, dt)


# ---------------------------------------------------------------- fiber ----

def test_fiber_arc_length_and_interpolation(curved_fiber):
    assert curved_fiber.length_m == pytest.approx(80 + 3 * np.pi)
    p = curved_fiber.point_at(10.0)
    assert np.allclose(p, [10.0, 0.0])
    s_back = curved_fiber.length_m - 10.0
    p2 = curved_fiber.point_at(s_back)
    assert np.allclose(p2, [10.0, 6.0], atol=1e-8)
    # tangent on the back-tracked leg points west
    t = curved_fiber.tangent_at(s_back)
    assert t[0] == pytest.approx(-1.0, abs=1e-9)
    s_out = curved_fiber.length_m + 1
    with pytest.raises(ValueError):
        curved_fiber.point_at(s_out)


def test_fiber_rejects_inconsistent_arc():
    with pytest.raises(ValueError):
        FiberPath(np.array([[0, 0], [0, 0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        FiberPath(np.array([[0, 0], [1, 0]]), np.array([0.0, 2.0]))


def test_channel_geometry_and_reversed_detection(curved_fiber):
    L = curved_fiber.length_m
    geom = ChannelGeometry(np.linspace(0, L, 90), curved_fiber)
    assert geom.positions.shape == (90, 2)
    rev = detect_reversed(geom)
    assert not rev[:38].any()
    assert rev[56:].all()           # second leg heads west
    # azimuth convention matches LocalFrame (east = 90 degrees)
    assert geom.azimuths_deg[0] == pytest.approx(90.0, abs=1e-9)
    assert geom.azimuths_deg[-1] == pytest.approx(270.0, abs=1e-9)


def test_break_detection(regular_fiber):
    s = np.r_[np.arange(0, 50, 2.0), np.arange(80, 101, 2.0)]
    geom = ChannelGeometry(s, regular_fiber)
    breaks = break_segments(geom, fiber_gap_m=10.0)
    assert len(breaks) == 1
    before, after, gap = breaks[0]
    assert after == before + 1 and gap == pytest.approx(32.0)


# ---------------------------------------------------------------- gather ---

def test_load_strain_rate_entry(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    g = load_strain_rate(np.zeros((geom.n_channels, 50)), geom, dt_s=0.001)
    assert g.units == 'strain_rate'
    assert g.n_channels == geom.n_channels
    with pytest.raises(ValueError):
        DASGather(np.zeros((3, 4, 5)), geom, 0.001)
    with pytest.raises(ValueError):
        DASGather(np.zeros((geom.n_channels + 1, 4)), geom, 0.001)
    assert g.times.size == 50


# ------------------------------------------------------------ sensitivity -

def test_gauge_average_weights_constant_field(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 1.0), regular_fiber)
    target = ChannelGeometry(np.array([25.0, 50.0]), regular_fiber)
    op = gauge_average_operator(geom, target, gauge_m=10.0)
    r0 = op.rows[0]
    assert r0.status is RowStatus.OK
    assert r0.indices.tolist() == list(range(20, 31))
    assert r0.weights.sum() == pytest.approx(1.0)
    # trapezoidal end weights of a 10 m window on 1 m grid
    assert r0.weights[0] == pytest.approx(0.05)
    assert r0.weights[-1] == pytest.approx(0.05)
    data = np.full((geom.n_channels, 3), 7.0)
    out = op.apply(data)
    assert np.allclose(out, 7.0)


def test_gauge_average_fractional_window(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    target = ChannelGeometry(np.array([25.0]), regular_fiber)
    op = gauge_average_operator(geom, target, gauge_m=10.0)
    row = op.rows[0]
    # window [20, 30] lands exactly on 2 m samples
    assert row.indices.tolist() == [10, 11, 12, 13, 14, 15]
    assert np.allclose(row.weights, [0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def test_gauge_average_linear_field(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 1.0), regular_fiber)
    target = ChannelGeometry(np.array([40.0, 60.0]), regular_fiber)
    op = gauge_average_operator(geom, target, gauge_m=20.0)
    x = geom.channel_s
    data = np.broadcast_to(x[:, None], (x.size, 2)).copy()
    out = op.apply(data)
    assert out[:, 0] == pytest.approx([40.0, 60.0])   # boxcar mean preserves mean


def test_gauge_window_refuses_to_cross_break(regular_fiber):
    s = np.r_[np.arange(0, 50, 2.0), np.arange(80, 101, 2.0)]
    geom = ChannelGeometry(s, regular_fiber)
    target = ChannelGeometry(np.array([60.0]), regular_fiber)
    op = gauge_average_operator(geom, target, gauge_m=10.0, gap_tolerance_m=5.0)
    assert op.rows[0].status is RowStatus.GAP_CROSSING
    assert np.isnan(op.apply(np.ones((geom.n_channels, 2)))).all()
    # window fully on one side is fine
    target_ok = ChannelGeometry(np.array([30.0]), regular_fiber)
    op_ok = gauge_average_operator(geom, target_ok, gauge_m=10.0, gap_tolerance_m=5.0)
    assert op_ok.rows[0].valid


def test_gauge_window_out_of_range(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    target = ChannelGeometry(np.array([1.0]), regular_fiber)
    op = gauge_average_operator(geom, target, gauge_m=10.0)
    assert op.rows[0].status is RowStatus.OUT_OF_RANGE


def test_gauge_difference_on_ramp(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 1.0), regular_fiber)
    target = ChannelGeometry(np.array([50.0]), regular_fiber)
    op = gauge_difference_operator(geom, target, gauge_m=10.0)
    # axial field u(s) = 0.01 s  ->  (u(s+L/2)-u(s-L/2))/L = 0.01
    data = (0.01 * geom.channel_s)[:, None]
    out = op.apply(data)
    assert out[0, 0] == pytest.approx(0.01, abs=1e-12)
    # weights sum to zero for a difference operator
    assert op.rows[0].weights.sum() == pytest.approx(0.0, abs=1e-12)


def test_reconstruct_channels_helper(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 1.0), regular_fiber)
    gather = make_gather(geom)
    target = ChannelGeometry(np.arange(10, 91, 5.0), regular_fiber)
    out, op = reconstruct_channels(gather, target, gauge_m=8.0)
    assert out.data.shape[0] == target.n_channels
    assert out.tags['gauge_m'] == '8'
    assert all(r.valid for r in op.rows)


def test_directional_sensitivity(regular_fiber):
    s = np.array([0.0, 25.0, 50.0])
    p = directional_sensitivity_p(regular_fiber, s, slowness_azimuth_deg=90.0)
    assert np.allclose(p, 1.0)                       # fibre parallel to P ray
    p_perp = directional_sensitivity_p(regular_fiber, s, slowness_azimuth_deg=0.0)
    assert np.allclose(p_perp, 0.0)                  # blind zone
    sh = directional_sensitivity_s(regular_fiber, s,
                                   slowness_azimuth_deg=90.0,
                                   polarization_azimuth_deg=0.0)
    assert np.allclose(sh, 0.0)                      # SH polarised across fibre


def test_axial_field_on_curved_fiber(curved_fiber):
    geom = ChannelGeometry(np.linspace(0, curved_fiber.length_m, 61), curved_fiber)
    # uniform 3C-ish field purely east
    fld = np.zeros((geom.n_channels, 2, 5))
    fld[:, 0, :] = 2.0
    axial = axial_field(fld, curved_fiber, geom.channel_s)
    assert np.allclose(axial[:24, 0], 2.0, atol=1e-8)
    assert np.allclose(axial[38:, 0], -2.0, atol=1e-8)   # back-tracked leg


# ----------------------------------------------------------------- quality -

def test_assess_quality_flags(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((geom.n_channels, 400))
    data[5] = 0.0                                    # dead/constant channel
    data[10] = np.sign(data[10]) * 5.0               # clipped/rail pile
    data[12] *= 1e-7                                 # very weak -> low quality
    gather = DASGather(data, geom, dt_s=0.002)
    qa = assess_quality(gather, rms_floor=1e-4,
                        saturated_fraction_limit=0.01)
    assert qa.flags[5] & QualityFlag.DEAD_CHANNEL
    assert qa.flags[10] & QualityFlag.SATURATED
    assert qa.flags[12] & QualityFlag.LOW_QUALITY
    kinds = {seg.kind for seg in qa.segments}
    assert {'dead-channel', 'saturated', 'low-quality'} <= kinds


def test_assess_quality_dead_fiber_and_reversed(curved_fiber):
    # introduce a break and dead downstream channels
    s = np.r_[np.arange(0, 40, 2.0), np.arange(79.4, curved_fiber.length_m + 0.01, 2.0)]
    geom = ChannelGeometry(s, curved_fiber)
    rng = np.random.default_rng(2)
    data = rng.standard_normal((geom.n_channels, 200))
    gather = DASGather(data, geom, dt_s=0.002)
    qa = assess_quality(gather, fiber_gap_m=10.0)
    assert (qa.flags & int(QualityFlag.DEAD_FIBER)).any()
    assert (qa.flags & int(QualityFlag.REVERSED)).any()
    assert any(seg.kind == 'dead-fiber' for seg in qa.segments)
    assert any(seg.kind == 'reversed' for seg in qa.segments)


# ---------------------------------------------------------------- resample -

def test_resample_spatial_preserves_linear_field_and_lineage(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    data = np.broadcast_to(geom.channel_s[:, None] * 0.5,
                           (geom.n_channels, 3)).copy()
    gather = DASGather(data, geom, dt_s=0.002)
    out, lineage = resample_spatial(gather, spacing_m=5.0)
    assert np.allclose(out.geometry.channel_s, np.arange(0, 101, 5.0))
    assert np.allclose(out.data[:, 0], out.geometry.channel_s * 0.5)
    # provenance of the first interior target: two bracketing originals
    info = lineage.source_channels_of(1)
    assert [cid for cid, _, _ in info] == [geom.channel_ids[2], geom.channel_ids[3]]
    weights = np.array([w for _, _, w in info])
    assert weights.sum() == pytest.approx(1.0)
    arc, pos = lineage.location_of(1)
    assert arc == pytest.approx(5.0)
    assert np.allclose(pos, [5.0, 0.0])
    # reverse lookup: original channel participates in nearby targets
    users = lineage.target_indices_from_source(3)
    assert users.size >= 1


def test_resample_spatial_does_not_span_gap(regular_fiber):
    s = np.r_[np.arange(0, 50, 2.0), np.arange(80, 101, 2.0)]
    geom = ChannelGeometry(s, regular_fiber)
    gather = DASGather(np.ones((geom.n_channels, 4)), geom, dt_s=0.002)
    out, lineage = resample_spatial(gather, spacing_m=5.0, gap_tolerance_m=10.0)
    target_s = out.geometry.channel_s
    gap_targets = (target_s > 50.0) & (target_s < 80.0)
    assert np.isnan(out.data[gap_targets]).all()
    assert all(lineage.samples[k].status is RowStatus.GAP_CROSSING
               for k in np.flatnonzero(gap_targets))
    out2, _ = resample_spatial(gather, spacing_m=5.0, gap_tolerance_m=10.0,
                               drop_invalid=True)
    assert ((out2.geometry.channel_s <= 50.0) |
            (out2.geometry.channel_s >= 80.0)).all()


def test_resample_time(regular_fiber):
    geom = ChannelGeometry(np.arange(0, 101, 2.0), regular_fiber)
    gather = DASGather(np.ones((geom.n_channels, 100)), geom, dt_s=0.002)
    out = resample_time(gather, 0.001)
    assert out.data.shape[1] == 200
    assert out.dt_s == 0.001
    # geometry/traceability untouched
    assert out.geometry.n_channels == geom.n_channels


# ---------------------------------------------------------------- pipeline -

def test_process_das_end_to_end(regular_fiber):
    rng = np.random.default_rng(3)
    s = np.arange(0, 101, 1.0)
    data = rng.standard_normal((s.size, 200))
    data[50] = 0.0                                  # dead channel
    cfg = DASConfig(gauge_m=10.0, output_spacing_m=5.0, output_dt_s=0.004,
                    rms_floor=1e-6)
    product = process_das(data, regular_fiber, s, dt_s=0.002, config=cfg)
    out = product.gather
    assert out.dt_s == 0.004
    assert np.allclose(out.geometry.channel_s, np.arange(0, 101, 5.0))
    # lineage round trip: target -> sources -> arc -> world position
    trace = product.trace_output(2)
    assert trace['arc_s'] == pytest.approx(10.0)
    assert len(trace['sources']) >= 2
    pos = product.trace_world_position(trace['sources'][0]['channel_id'])
    assert pos.shape == (2,)
    # targets whose 10 m gauge window covers the dead channel are invalidated
    invalid_s = [row.center_s for row in product.operator.rows
                 if not row.valid]
    assert any(45.0 <= s <= 55.0 for s in invalid_s)
    assert np.isnan(product.gather.data[
        [k for k, s in enumerate(product.target_geometry.channel_s)
         if 45.0 <= s <= 55.0]]).all()
    # resampled time axis
    assert out.n_samples == int(round(200 * 0.002 / 0.004))


def test_process_das_difference_mode(regular_fiber):
    s = np.arange(0, 101, 1.0)
    data = np.broadcast_to((0.02 * s)[:, None], (s.size, 50)).copy()
    cfg = DASConfig(gauge_m=10.0, mode='difference')
    product = process_das(data, regular_fiber, s, dt_s=0.002, config=cfg)
    interior = slice(2, -2)
    assert np.nanmean(product.gather.data[interior]) == pytest.approx(0.02, abs=1e-10)


def test_csv_io_roundtrip_and_id_traceability(tmp_path, regular_fiber):
    s = np.arange(0, 101, 2.0)
    fiber_csv = tmp_path / 'fiber.csv'
    fiber_csv.write_text(
        'easting,northing,elev\n' +
        '\n'.join(f'{x:.3f},0.0,0.0' for x in regular_fiber.points[:, 0]))
    chan_csv = tmp_path / 'channels.csv'
    chan_csv.write_text(
        'channel_id,arc_s\n' +
        '\n'.join(f'cable-1/{i:03d},{v:.3f}' for i, v in enumerate(s)))
    np.save(tmp_path / 'sr.npy', np.zeros((s.size, 10)))

    fiber = read_fiber_csv(fiber_csv)
    assert fiber.length_m == pytest.approx(100.0)
    geom = read_channel_csv(chan_csv, fiber)
    assert geom.channel_ids[3] == 'cable-1/003'
    gather = read_strain_rate_npy(tmp_path / 'sr.npy', geom, dt_s=0.002)
    assert gather.n_channels == s.size

    rng = np.random.default_rng(5)
    cfg = DASConfig(gauge_m=4.0, output_spacing_m=10.0, fiber_gap_m=9.0)
    prod = process_das(rng.standard_normal((s.size, 100)), fiber, s,
                       dt_s=0.002, config=cfg, channel_ids=geom.channel_ids)
    # output ids are anchored to nearest original ids
    assert all(tid.endswith('#rg') for tid in prod.target_geometry.channel_ids)
    trace = prod.trace_output(1)
    assert trace['sources'][0]['channel_id'].startswith('cable-1/')
    pos = prod.trace_world_position('cable-1/010')
    assert pos[0] == pytest.approx(20.0)
