import abc
import numpy
from typing import Tuple, Dict


class Device(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_usage_perc(self, timeslice: Tuple[int, int]) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def get_qd(self, timeslice: Tuple[int, int]) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def get_err_count(self, timeslice: Tuple[int, int]) -> numpy.ndarray:
        pass


class NetAdapter(Device):
    def __init__(self, node: str, name: str, max_speed: int, duplex: bool = True) -> None:
        self.timestamps = None  # type: numpy.ndarray
        self.tss = None  # type: Dict[str, numpy.ndarray]

    def set_load(self, timestamps: numpy.ndarray, **tss: numpy.ndarray) -> None:
        pass

    def get_usage_perc(self, timeslice: Tuple[int, int]) -> numpy.ndarray:
        return get_slice(self.timestamps, self.tss[''])
