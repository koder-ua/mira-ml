from typing import Callable, List, Tuple

import numpy
from scipy.spatial import distance
from sklearn.cluster import KMeans


from .data import Node, NodeHistoFunc, HDistFunc, Compl


def cluster_vms(vms: List[Node], histo_func: Callable[[Node], numpy.ndarray]) -> List[List[float]]:
    # ds = DBSCAN(eps=0.05)
    # ds = SpectralClustering(affinity='precomputed', n_clusters=3)
    # for vid, lb in sorted(zip(vm_ids, ds.fit_predict(dists2d)), key=lambda x: x[1]):
    #     print(vid, lb)

    histos = list(map(histo_func, vms))
    ds = KMeans(n_clusters=3)
    return ds.fit_predict(histos)


def get_metric_slice(dev: str, metric: str, start_tm: float, stop_tm: float) -> NodeHistoFunc:
    def closure(vm: Node) -> numpy.ndarray:
        mtr = vm.metric(dev, metric)
        items_per_time_unit = len(mtr) / (vm.stop_time - vm.start_time)
        idx1 = int((start_tm - vm.start_time) * items_per_time_unit)
        idx2 = int((vm.stop_time - stop_tm) * items_per_time_unit)
        if idx2 == 0:
            return mtr[idx1:]
        else:
            return mtr[idx1:-idx2]
    return closure


def metric_distance(metr_func: NodeHistoFunc, distance: HDistFunc) -> Callable[[Node, Node], float]:
    def closure(vm1: Node, vm2: Node) -> float:
        mtr1 = metr_func(vm1)
        mtr2 = metr_func(vm2)
        assert abs(len(mtr1) - len(mtr2)) <= 2
        assert len(mtr1) != 0
        assert len(mtr2) != 0

        if len(mtr1) > len(mtr2):
            mtr1 = mtr1[:len(mtr2)]
        else:
            mtr2 = mtr2[:len(mtr1)]

        return distance(mtr1, mtr2) / len(mtr1)
    return closure


def moving_average(data: numpy.ndarray, n: int) -> numpy.ndarray:
    ret = numpy.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_cross_vm_dist(vms: List[Node], dist_func: Callable[[Node, Node], float], sort: bool = False) -> numpy.ndarray:
    assert len(vms) > 1

    if sort:
        vms = [vms[0]] + sorted(vms[1:], key=lambda vm: dist_func(vms[0], vm))

    res = numpy.zeros(shape=(len(vms), len(vms)), dtype=float)
    for idx, vm1 in enumerate(vms):
        for idx2, vm2 in enumerate(vms[idx + 1:], idx + 1):
            res[idx, idx2] = res[idx2, idx] = dist_func(vm1, vm2)
    return res


def get_cross_dist(vecs1: List[numpy.ndarray],
                   vecs2: List[numpy.ndarray],
                   dist_func: Callable[[numpy.ndarray, numpy.ndarray], float], sort: bool = False) -> numpy.ndarray:
    if sort:
        vecs2 = sorted(vecs2, key=lambda vec: dist_func(vecs1[0], vec))

    res = numpy.zeros(shape=(len(vecs1), len(vecs2)), dtype=float)
    for idx, vec1 in enumerate(vecs1):
        for idx2, vec2 in enumerate(vecs2):
            res[idx, idx2] = dist_func(vec1, vec2)
    return res


def complimentary_metric(vec1: numpy.ndarray, vec2: numpy.ndarray, max_val: float) -> Compl:
    """
    Return three number, which show how complimentary vm profiles is, and
    part of time, when summary profile was over max_val value and
    utilization which is equal to 1.0 if system if fully loaded(or overloaded) all the time
    """
    assert len(vec1) == len(vec2)
    total_load = vec1 + vec2
    unused = max_val - total_load
    total_capacity = max_val * len(vec1)

    overload = -unused[unused < 0].sum() / total_capacity
    utilization = numpy.clip(unused, 0, None, unused).sum() / total_capacity
    complimentarity = total_load.std() / total_capacity

    return Compl(complimentarity, overload, utilization)


def sort_by_hdistance(vm: Node, vms: List[Node], histo_func: NodeHistoFunc,
                      dist_func: HDistFunc = distance.euclidean) -> Tuple[List[Node], List[float]]:
    base_h = histo_func(vm)
    distances = [(dist_func(base_h, histo_func(cvm)), cvm) for cvm in vms]
    distances.sort()
    dists, svms = zip(*distances)
    return list(svms), list(dists)


def pair_distances(vms: List[Node], histo_func: NodeHistoFunc, dist_func: HDistFunc = distance.euclidean) -> numpy.ndarray:

    res = numpy.zeros(shape=(len(vms), len(vms)), dtype=float)
    histos = list(map(histo_func, vms))

    for idx, h1 in enumerate(histos):
        for idx2, h2 in enumerate(histos[idx + 1:], idx + 1):
            res[idx, idx2] = res[idx2, idx] = dist_func(h1, h2)

    return res

