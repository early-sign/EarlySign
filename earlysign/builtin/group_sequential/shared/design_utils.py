import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.builtin.group_sequential.adapters import protocol as adapter


def get_standardized_drift(protocol: GST.Protocol) -> float:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_standardized_drift"""
    return adapter.get_standardized_drift(protocol)


def get_info_times(schedule: GST.ScheduleSpec) -> np.ndarray:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_info_times"""
    return adapter.get_info_times(schedule)


def get_final_efficacy_boundary(protocol: GST.Protocol) -> float:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_final_efficacy_boundary"""
    return adapter.get_final_efficacy_boundary(protocol)
