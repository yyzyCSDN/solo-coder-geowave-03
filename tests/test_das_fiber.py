import numpy as np
import pytest
from geowave.das import (
    FiberGeometry, build_das_gather, process_das, DasProcessConfig,
    reconstruct_gauge, correct_sensitivity, mark_quality,
    resample_spatial, resample_temporal, apply_gauge, as_strain_rate,
    directional_sensitivity, resample_channels, target_grid,
)


def curved_u_fiber():
    """Straight run north, a 90-degree bend, then a reversed run back south."""
    ids = [f'c{i:03d}' for i in range(20)]
    x = [0.0] * 10 + [10.0] * 10
    y = [10.0 * i for i in range(10)] + [90.0 - 10.0 * i for i in range(10)]
    return FiberGeometry.build(ids, x, y)


def straight_fiber(n=21, d=2.0):
    ids = [f'c{i:03d}' for i in range(n)]
    return FiberGeometry.build(ids, np.zeros(n), d * np.arange(n))


# ------------------------------------------------------------- fiber geometry
def test_fiber_channel_positions_and_tangents():
    geo = curved_u_fiber()
    assert geo.n_channels == 20
    assert geo.location('c000') == (0.0, 0.0, 0.0)
    assert geo.location('c009') == (0.0, 90.0, 0.0)
    assert geo.arclength_m[5] == pytest.approx(50.0)
    north = geo.tangent[5]
    assert north[1] == pytest.approx(1.0)
    south = geo.tangent[15]
    assert south[1] == pytest.approx(-1.0)


def test_bend_is_detected_as_a_turn():
    geo = curved_u_fiber()
    angles = {t.index: t.angle_deg for t in geo.turns}
    assert any(abs(idx - 10) <= 1 and ang > 60.0 for idx, ang in angles.items())


def test_return_leg_is_marked_reversed():
    geo = curved_u_fiber()
    reversed_ids = set()
    for sec in geo.reversed_sections:
        reversed_ids.update(geo.channel_ids[sec.start_index:sec.stop_index + 1])
    assert 'c005' not in reversed_ids          # northbound leg follows reference
    assert {'c011', 'c019'}.issubset(reversed_ids)
    assert geo.is_reversed('c015')
    assert not geo.is_reversed('c003')


def test_gap_creates_break_and_resets_arclength_and_segments():
    ids = ['a', 'b', 'c', 'd', 'e']
    x = np.zeros(5)
    y = np.array([0.0, 10.0, 20.0, 200.0, 210.0])
    geo = FiberGeometry.build(ids, x, y, max_step_m=50.0)
    assert len(geo.breaks) == 1
    assert geo.breaks[0].after_index == 2
    assert geo.breaks[0].reason == 'gap'
    assert geo.segment_of('c') == 0
    assert geo.segment_of('d') == 1
    assert geo.arclength_m[3] == pytest.approx(0.0)
    assert geo.arclength_m[4] == pytest.approx(10.0)


def test_declared_break():
    geo = straight_fiber(n=6, d=5.0)
    geo2 = FiberGeometry.build(geo.channel_ids, geo.x, geo.y,
                               declared_breaks=[2])
    assert [b.after_index for b in geo2.breaks] == [2]
    assert geo2.segment_count == 2


# ------------------------------------------------------- strain-rate / gauge
def test_as_strain_rate_units():
    dt = 0.002
    t = np.arange(100) * dt
    phase = np.tile(3.0 * t, (2, 1))
    sr = as_strain_rate(phase, dt, input_kind='phase', phase_to_strain_scale=1.0)
    assert np.allclose(sr[:, 5:95], 3.0)
    strain = np.tile(2.0 * t, (1, 1))
    assert np.allclose(as_strain_rate(strain, dt, input_kind='strain')[:, 5:95], 2.0)


def test_gauge_integration_recovers_linear_strain_rate():
    geo = straight_fiber(n=21, d=2.0)               # s = 0..40 m
    s = geo.arclength_m
    n_t = 8
    # strain rate linear along the fiber: trapezoid gauge mean must be exact
    data = np.tile((1.5 + 0.25 * s)[:, None], (1, n_t))
    res = apply_gauge(data, geo, gauge_length_m=10.0)
    interior = [w for w in res.windows if not w.partial]
    assert interior
    for w in interior:
        k = res.windows.index(w)
        assert w.weights.sum() == pytest.approx(1.0)
        assert res.data[k, 0] == pytest.approx(1.5 + 0.25 * w.center_s_m)


def test_gauge_window_does_not_cross_break_and_flags_partial():
    ids = [f'c{i}' for i in range(8)]
    y = [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0]
    geo = FiberGeometry.build(ids, np.zeros(8), y, declared_breaks=[3])
    data = np.ones((8, 4))
    res = apply_gauge(data, geo, gauge_length_m=20.0, min_coverage=0.25)
    for w in res.windows:
        segs = {int(geo.segments[i]) for i in w.indices}
        assert len(segs) == 1                        # never mixes fiber segments
    partials = [w for w in res.windows if w.partial]
    assert partials


# ----------------------------------------------------- directional sensitivity
def test_directional_sensitivity_cos_squared():
    geo = straight_fiber(n=3, d=10.0)                # tangent along +y
    direction = np.array([[np.sin(np.pi / 3), np.cos(np.pi / 3), 0.0]] * 3)
    sens, angle = directional_sensitivity(geo, directions=direction, mode='cos2')
    assert np.allclose(sens, 0.25)
    assert np.allclose(angle, 60.0)


def test_broadside_channels_are_masked_or_flagged():
    geo = straight_fiber(n=3, d=10.0)
    broadside = np.array([[1.0, 0.0, 0.0]] * 3)      # 90 deg to the fiber
    gather = build_das_gather(np.ones((3, 10)), geo, 0.001)
    out = correct_sensitivity(gather, directions=broadside, action='correct',
                              min_sensitivity=0.05)
    assert np.all(np.isnan(out.data))
    assert bool(out.directional.weak_mask.all())
    flagged = correct_sensitivity(gather, directions=broadside, action='flag')
    assert np.allclose(flagged.data, 1.0)
    assert bool(flagged.directional.weak_mask.all())


# --------------------------------------------------------------------- quality
def test_quality_flags_dead_clipped_weak_reversed_and_near_break():
    geo = curved_u_fiber()
    n_t = 200
    data = 0.1 * np.random.default_rng(0).standard_normal((20, n_t))
    data[3] = 0.0                                   # dead / flat
    data[7, :] = 5.0 * np.where(np.arange(n_t) % 2 == 0, 1.0, -1.0)  # clipped
    data[12] *= 1e-4                                # weak (and on reversed leg)
    table = mark_quality(build_das_gather(data, geo, 0.001)).quality
    flags = table.channel_flags()
    assert 'dead' in flags['c003']
    assert 'clipped' in flags['c007']
    assert 'low-signal' in flags['c012']
    assert 'reversed' in flags['c018']
    assert 'near-break' in flags['c000']
    assert table.low_quality_zones
    assert len(table.breaks) == 0


# ------------------------------------------------------------------ resampling
def test_spatial_resampling_keeps_channel_to_position_traceability():
    geo = straight_fiber(n=11, d=10.0)              # s = 0..100
    n_t = 4
    data = np.tile(geo.arclength_m[:, None], (1, n_t))
    out = resample_channels(data, geo, spacing_m=5.0, method='linear')
    new_geo = out.geometry
    assert new_geo.n_channels == 21
    # channel at s = 55 m mixes original c005/c006 50/50 and sits at (0, 55)
    k = 11
    origins = dict(out.resampling.provenance[k].sources)
    assert origins == pytest.approx({'c005': 0.5, 'c006': 0.5})
    p = out.resampling.provenance[k]
    assert (p.x, p.y) == pytest.approx((0.0, 55.0))
    assert out.data[k, 0] == pytest.approx(55.0)


def test_resampling_never_crosses_breaks():
    ids = [f'c{i}' for i in range(8)]
    y = [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0]
    geo = FiberGeometry.build(ids, np.zeros(8), y, declared_breaks=[3])
    data = np.arange(8, dtype=float)[:, None] * np.ones((1, 3))
    out = resample_channels(data, geo, spacing_m=10.0, method='linear')
    for prov in out.resampling.provenance:
        source_segments = {geo.segment_of(cid) for cid, _ in prov.sources}
        assert len(source_segments) <= 1


def test_gather_provenance_chain_reaches_original_channels():
    geo = straight_fiber(n=11, d=10.0)
    data = np.tile(geo.arclength_m[:, None], (1, 5))
    g0 = build_das_gather(data, geo, 0.002)
    g1 = reconstruct_gauge(g0, gauge_length_m=20.0)
    g2 = resample_spatial(g1, spacing_m=5.0)
    new_id = g2.channel_ids[3]
    origins = g2.origins_of(new_id)
    assert origins
    assert all(cid in geo.channel_ids for cid, w in origins)
    assert sum(w for _, w in origins) == pytest.approx(1.0)
    chain = g2.trace_to_origin(new_id)
    assert chain[0].startswith('strain-rate:das-')
    assert chain[-1].split(':')[-1] in geo.channel_ids
    assert len(chain) == 3                           # raw -> gauge -> resampled


def test_temporal_resampling_preserves_channel_provenance():
    geo = straight_fiber(n=6, d=5.0)
    g0 = build_das_gather(np.random.default_rng(1).standard_normal((6, 100)),
                          geo, 0.004)
    g1 = resample_spatial(g0, spacing_m=2.5)
    g2 = resample_temporal(g1, 0.002)
    assert g2.n_samples == 200
    assert g2.channel_ids == g1.channel_ids
    assert g2.origins_of(g2.channel_ids[2]) == g1.origins_of(g1.channel_ids[2])


# -------------------------------------------------------------------- pipeline
def test_process_das_end_to_end_on_curved_fiber():
    geo = curved_u_fiber()
    n_t = 128
    t = np.arange(n_t) * 0.002
    data = np.zeros((20, n_t))
    for i in range(20):
        s = geo.arclength_m[i]
        data[i] = np.sin(2 * np.pi * 40 * t - 0.2 * s) * (0.5 + 0.01 * i)
    cfg = DasProcessConfig(
        gauge_length_m=10.0, spatial_spacing_m=5.0, new_dt_s=0.001,
        source_xyz=(0.0, 1000.0, 0.0))
    result = process_das(data, geo, 0.002, cfg)
    labels = ' | '.join(result.history)
    assert 'build' in labels and 'gauge' in labels
    assert 'resample-space' in labels and 'resample-time' in labels
    assert 'directional' in labels and 'quality' in labels
    assert result.dt_s == pytest.approx(0.001)
    assert result.quality is not None
    assert result.parent is not None
    # new channels still trace back to the original c000..c019 set
    origins = result.origins_of(result.channel_ids[5])
    assert origins
    assert set(geo.channel_ids).issuperset(cid for cid, _ in origins)
    assert sum(w for _, w in origins) == pytest.approx(1.0)
    # reversed run information survives the resampled geometry
    assert any('reversed' in tags for tags in result.quality.channel_flags().values())
