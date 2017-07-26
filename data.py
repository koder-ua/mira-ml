import time
from typing import NamedTuple, Callable, List, Union, Dict, cast

from cephlib.units import b2ssize_10
import numpy


Compl = NamedTuple('Compl', [('complimentarity', float), ('overload', float), ('utilization', float)])
TSDBCreds = NamedTuple("TSDBCreds", [("user", str), ("password", str), ("database", str), ("host", str),
                                     ('port', int)])
ArrTransformer = Callable[[numpy.ndarray], numpy.ndarray]
AnyNumArr = Union[numpy.ndarray, List[Union[float, int]]]
HDistFunc = Callable[[numpy.ndarray, numpy.ndarray], float]

class Metric:
    def get_histo(self):
        pass


class TSStatInfo:
    def __init__(self):
        self.mean = None  # type: float
        self.median = None  # type: float
        self.stddev = None  # type: float
        self.data = None  # type: numpy.ndarray
        self.units = None  # type: str
        self.total = None  # type: float

    def __str__(self):
        return "{}~{}{} => {}{}".format(b2ssize_10(self.mean), b2ssize_10(self.median), self.units,
                                        b2ssize_10(self.total), self.units)


class VM:
    def __init__(self, vm_id: str) -> None:
        self.vm_id = vm_id
        self.metrics = {}  # type: Dict[str, Dict[str, TSStatInfo]]
        self.vm_id_short = vm_id[:6]
        self.start_time = None
        self.stop_time = None

    def __lt__(self, vm):
        return self.vm_id < vm.vm_id

    @property
    def lifetime(self) -> float:
        return self.stop_time - self.start_time

    def __str__(self) -> str:
        return "VM({})".format(self.vm_id)

    @property
    def full_info(self) -> str:
        res = "VM({})".format(self.vm_id)
        step = "    "
        res += "\n" + step + "Lifetime: {:.1f}H".format(self.lifetime / 3600)
        for metric, data in sorted(self.metrics.items()):
            res += "\n" + step + metric
            for dev, stat in sorted(data.items()):
                res += "\n" + step * 2 + "{}: {!s}".format(dev, stat)
        return res


VMHistoFunc = Callable[[VM], numpy.ndarray]


class Metric1D:
    device = None
    metric = None
    histo_bins = None
    filters = None

    @classmethod
    def get_histo(cls):
        return histo1d(cls.device, cls.metric, cls.histo_bins, filters=cls.filters)


class Metric2D(Metric1D):
    metric2 = None
    histo_bins2 = None

    @classmethod
    def get_histo(cls):
        return histo2d(cls.device, cls.metric, cls.metric2, cls.histo_bins, cls.histo_bins2,
                       filters=cls.filters)


def cut1d_r(r: int = 2) -> ArrTransformer:
    def closure(arr: numpy.ndarray) -> numpy.ndarray:
        arr[:r] = 0
        return arr
    return closure


def cut2d_r(r: int = 2) -> ArrTransformer:
    def closure(arr: numpy.ndarray) -> numpy.ndarray:
        arr[:r, :r] = 0
        return arr
    return closure


def blur1d(bc: float = 0.2) -> ArrTransformer:
    def closure(arr: numpy.ndarray) -> numpy.ndarray:
        b_arr = arr * (1 - 2 * bc)
        b_arr[:-1] += arr[1:] * bc
        b_arr[1:] += arr[:-1] * bc
        return b_arr
    return closure


def blur2d(bc: float = 0.058579) -> ArrTransformer:
    def closure(arr: numpy.ndarray) -> numpy.ndarray:
        b_arr = arr * (1 - (4 + 4 / 2 ** 0.5) * bc)

        b_arr[:-1, :] += arr[1:, :] * bc
        b_arr[1:, :] += arr[:-1, :] * bc
        b_arr[:, :-1] += arr[:, 1:] * bc
        b_arr[:, 1:] += arr[:, :-1] * bc

        bc2 = bc / 2 ** 0.5
        b_arr[1:, 1:] += arr[:-1, :-1] * bc2
        b_arr[:-1, 1:] += arr[1:, :-1] * bc2
        b_arr[1:, :-1] += arr[:-1, 1:] * bc2
        b_arr[:-1, :-1] += arr[1:, 1:] * bc2

        return b_arr
    return closure


class CPUMetrics(Metric1D):
    device = 'None'
    metric = 'virt_cpu_time'
    histo_bins = numpy.linspace(0, 10**10, 20)
    filters = [cut1d_r(2), blur1d()]


class IOMetricsWriteVda(Metric1D):
    device = 'vda'
    metric = 'virt_disk_ops_write'
    histo_bins = numpy.linspace(0, 400, 20)
    filters = [cut1d_r(2), blur1d()]


class IOMetricsWriteWszVda(Metric2D, IOMetricsWriteVda):
    metric2 = 'w_size'
    histo_bins2 = numpy.linspace(0, 600, 20)
    filters = [cut2d_r(2), blur2d()]


def histo1d(device: str, metric: str, bins: numpy.ndarray,
            norm: bool = True,
            filters: List[ArrTransformer] = None) -> VMHistoFunc:
    def closure(vm: VM) -> numpy.ndarray:
        data = vm.metrics[device][metric].copy()
        numpy.clip(data, bins[0], bins[-1], data)
        histo = numpy.histogram(data, bins)[0]

        if filters:
            for flt in cast(List[ArrTransformer], filters):
                histo = flt(histo)

        if norm:
            return histo / histo.sum()

        return histo
    return closure


def histo2d(device: str, metric1: str, metric2: str,
            bins1: numpy.ndarray, bins2: numpy.ndarray,
            norm: bool = True,
            filters: List[ArrTransformer] = None) -> VMHistoFunc:
    def closure(vm: VM) -> numpy.ndarray:
        x = vm.metrics[device][metric1].copy()
        y = vm.metrics[device][metric2].copy()
        numpy.clip(x, bins1[0], bins1[-1], x)
        numpy.clip(y, bins2[0], bins2[-1], y)
        histo = numpy.histogram2d(x, y, bins=(bins1, bins2))[0]

        if filters:
            for flt in cast(List[ArrTransformer], filters):
                histo = flt(histo)

        if norm:
            histo /= histo.sum()

        return histo.flatten()
    return closure


def influxtime2tm(val):
    return time.mktime(time.strptime(val, "%Y-%m-%dT%H:%M:%S.%fZ" if '.' in val else "%Y-%m-%dT%H:%M:%SZ"))


def add_iop_size(vm: VM, device: str) -> None:
    per_dev = vm.metrics[device]
    oct_write = per_dev['virt_disk_octets_write']
    oct_read = per_dev['virt_disk_octets_read']
    ops_write = per_dev['virt_disk_ops_write']
    ops_read = per_dev['virt_disk_ops_read']

    w_ok = ops_write > 1.0
    r_ok = ops_read > 1.0

    w_size = numpy.zeros(oct_write.shape, dtype=float)
    r_size = numpy.zeros(oct_read.shape, dtype=float)

    w_size[w_ok] = oct_write[w_ok] / ops_write[w_ok]
    r_size[r_ok] = oct_read[r_ok] / ops_read[r_ok]

    per_dev['w_size'] = w_size
    per_dev['r_size'] = r_size


def parse_db_creds(creds:str) -> TSDBCreds:
    # user:passwd@host:[port]/db
    creds, db = creds.rsplit("/", 1)
    user_passwd, host_port = creds.rsplit("@", 1)
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host = host_port
        port = None
    user, passwd = user_passwd.split(":", 1)
    return TSDBCreds(user, passwd, db, host, port)
