import numpy as np

from earlysign.builtin.group_sequential.adapters import protocol as adapter
from earlysign.builtin.group_sequential.schema.protocol import Protocol
from earlysign.builtin.group_sequential.schema.timers import ScheduleSpec


def get_standardized_drift(protocol: Protocol) -> float:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_standardized_drift"""
    return adapter.get_standardized_drift(protocol)


def get_info_times(schedule: ScheduleSpec) -> np.ndarray:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_info_times"""
    return adapter.get_info_times(schedule)


def get_final_efficacy_boundary(protocol: Protocol) -> float:
    """DEPRECATED: Use earlysign.builtin.group_sequential.adapters.protocol.get_final_efficacy_boundary"""
    return adapter.get_final_efficacy_boundary(protocol)
