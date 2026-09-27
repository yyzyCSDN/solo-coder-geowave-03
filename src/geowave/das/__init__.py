"""Distributed Acoustic Sensing (DAS) along curved optical fibre.

Channels are never treated as ordinary receivers: every channel is bound to an
arc-length position on a :class:`FiberPath`, strain-rate traces are rebuilt with
an explicit gauge length and cable-direction sensitivity, broken / reversed /
low-quality sections are flagged, and resampling preserves original-channel to
world-position provenance.
"""
from .fiber import (FiberPath, ChannelGeometry, detect_reversed, hairpin_indices)
from .gather import DASGather, load_strain_rate
from .sensitivity import (ReconstructionOperator, OperatorRow, RowStatus,
                          gauge_average_operator, gauge_difference_operator,
                          reconstruct_channels, axial_projection, axial_field,
                          directional_sensitivity_p, directional_sensitivity_s)
from .quality import (QualityFlag, QualityAssessment, ChannelQC, Segment,
                      assess_quality, apply_flags, break_segments,
                      contiguous_segments)
from .resample import (resample_spatial, resample_time, SpatialLineage,
                       SpatialSample, lineage_from_operator)
from .pipeline import DASConfig, DASProduct, process_das
from .io import (read_fiber_csv, read_channel_csv, read_strain_rate_npy,
                 save_strain_rate_npy)

__all__ = [
    'FiberPath', 'ChannelGeometry', 'detect_reversed', 'hairpin_indices',
    'DASGather', 'load_strain_rate',
    'ReconstructionOperator', 'OperatorRow', 'RowStatus',
    'gauge_average_operator', 'gauge_difference_operator',
    'reconstruct_channels', 'axial_projection', 'axial_field',
    'directional_sensitivity_p', 'directional_sensitivity_s',
    'QualityFlag', 'QualityAssessment', 'ChannelQC', 'Segment',
    'assess_quality', 'apply_flags', 'break_segments', 'contiguous_segments',
    'resample_spatial', 'resample_time', 'SpatialLineage', 'SpatialSample',
    'lineage_from_operator',
    'DASConfig', 'DASProduct', 'process_das',
    'read_fiber_csv', 'read_channel_csv', 'read_strain_rate_npy',
    'save_strain_rate_npy',
]
