"""Distributed Acoustic Sensing (DAS) along curved fiber.

Unlike ordinary receiver gathers, DAS channels must be handled through their
fiber geometry:

- :class:`FiberGeometry` maps channels to 3-D positions, arc length, tangent
  vectors, segments, breaks and reversed sections.
- Strain-rate channels are reconstructed by gauge-length integration
  (:func:`~geowave.das.gauge.apply_gauge`) and corrected for directional
  sensitivity (:mod:`geowave.das.sensitivity`).
- Breaks, reversed sections and low-quality zones are marked
  (:func:`~geowave.das.qc.assess_channels`).
- Spatial and temporal resampling keep every new channel traceable to the
  original channels and their positions
  (:mod:`geowave.das.resample`, :func:`~geowave.das.gather.process_das`).
"""
from .geometry import (
    FiberGeometry, FiberBreak, ReversedSection, FiberTurn, load_fiber_geometry,
)
from .gauge import as_strain_rate, apply_gauge, gauge_windows, GaugeResult, GaugeWindow
from .sensitivity import (
    directional_sensitivity, apply_directional, incidence_directions,
    DirectionalResult,
)
from .qc import (
    assess_channels, QualityTable, ChannelQuality, QualityZone,
)
from .resample import (
    resample_channels, resample_time, target_grid,
    ResampledDas, SpatialResampling, ChannelProvenance,
)
from .gather import (
    DasGather, DasProcessConfig, build_das_gather, process_das,
    reconstruct_gauge, correct_sensitivity, mark_quality,
    resample_spatial, resample_temporal,
)

__all__ = [
    'FiberGeometry', 'FiberBreak', 'ReversedSection', 'FiberTurn',
    'load_fiber_geometry',
    'as_strain_rate', 'apply_gauge', 'gauge_windows', 'GaugeResult', 'GaugeWindow',
    'directional_sensitivity', 'apply_directional', 'incidence_directions',
    'DirectionalResult',
    'assess_channels', 'QualityTable', 'ChannelQuality', 'QualityZone',
    'resample_channels', 'resample_time', 'target_grid',
    'ResampledDas', 'SpatialResampling', 'ChannelProvenance',
    'DasGather', 'DasProcessConfig', 'build_das_gather', 'process_das',
    'reconstruct_gauge', 'correct_sensitivity', 'mark_quality',
    'resample_spatial', 'resample_temporal',
]
