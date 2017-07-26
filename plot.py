from typing import List, Callable, Iterable


import numpy
from matplotlib.axes import Axes


from .data import VM, VMHistoFunc
from .ml import moving_average


def plot_io(vms: List[VM], dists: List[float], ax_func: Callable[[int], Axes], device: str) -> None:

    for idx, (dist, vm) in enumerate(zip(dists, vms)):
        per_dev = vm.metrics[device]
        oct_write = per_dev['virt_disk_octets_write']
        oct_read = per_dev['virt_disk_octets_read']
        ops_write = per_dev['virt_disk_ops_write']
        ops_read = per_dev['virt_disk_ops_read']

        oct_write = oct_write[ops_write > 1.0]
        ops_write = ops_write[ops_write > 1.0]
        oct_read = oct_read[ops_read > 1.0]
        ops_read = ops_read[ops_read > 1.0]
        w_block_sz_kb = oct_write / ops_write / 1024
        r_block_sz_kb = oct_read / ops_read / 1024

        ax = ax_func(idx)
        ax.plot(w_block_sz_kb, ops_write, '.', label=vm.vm_id_short, color="#FF606040")
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 500)

        ax.set_title("{}   {:.5f}".format(vm.vm_id_short, dist))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



def plot_histos(vms: List[VM], histograms: Iterable[numpy.ndarray],
                dists: List[float], ax_func: Callable[[int], Axes]) -> None:
    for idx, (dist, histo, vm) in enumerate(zip(dists, histograms, vms)):
        ax = ax_func(idx)
        ax.bar(numpy.arange(len(histo)) + 0.5, histo)
        ax.set_title("{}   {:.5f}".format(vm.vm_id_short, dist))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_vm_lifetime(vms: List[VM],  ax: Axes) -> None:
    for idx, vm in enumerate(vms):
        ax.plot([vm.start_time, vm.stop_time], [idx, idx], '-', linewidth=2)
        ax.annotate(vm.vm_id[:6],
                    xy=(vm.stop_time, idx),
                    xytext=(vm.stop_time * 1.00001, idx - 0.1), size='x-large')
        ax.grid(True)

def plot_cross_vm_dist(dist_matr: numpy.ndarray, vm_labels: List[str], ax: Axes) -> None:
    ax.matshow(dist_matr)
    ax.set_xticks(numpy.arange(len(vm_labels)))
    ax.set_xticklabels(vm_labels, rotation=30, size='x-large')
    ax.set_yticks(numpy.arange(len(vm_labels)))
    ax.set_yticklabels(vm_labels, size='x-large')


def group_vms(vms: List[VM], histo_func: VMHistoFunc, ax: Axes) -> None:
    model = TSNE(n_components=2, random_state=12, verbose=True)
    histos = list(map(histo_func, vms))
    res = model.fit_transform(histos)
    for vm, (x, y) in zip(vms, res):
        ax.plot([x], [y], 'o')
        ax.annotate(vm.vm_id[:6], xy=(x, y), xytext=(x, y), size='x-large')


def plot_metrics(vms: List[VM], metr_func: VMHistoFunc, ax: Axes) -> None:
    for vm in vms:
        metr = metr_func(vm)
        metr = moving_average(metr, 90) # 15 min average
        ax.plot(metr, label=vm.vm_id_short)
        ax.legend()
