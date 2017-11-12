import sys
import os.path
import argparse
import itertools
from typing import List, Dict

import yaml
from scipy.spatial import distance
from matplotlib import pyplot
import matplotlib.gridspec as gridspec

from .data import IOMetricsWriteVda, add_iop_size, TSDBCreds, parse_db_creds, CPUMetrics, disk_metric
from .db import LocalDBNP, load_nodes, LocalDB
from .ml import get_metric_slice, metric_distance, sort_by_hdistance, get_cross_vm_dist, get_cross_dist, moving_average
from .plot import plot_cross_vm_dist, plot_metrics, plot_vm_lifetime, plot_cross_dist
from .influx_exporter import copy_data_from_influx, lookup_vms, connect_to_ts_database, most_loaded_vms


class Config:
    instances = None  # type: List[str]
    sqlite3db = None  # type: str
    sqlite_cls = None  # type: str
    ts_db = None  # type: str
    sync_metrics = None  # type: Dict[str, Dict[str, str]]


def connect_to_sqlite(cfg: Config) -> LocalDB:
    clss = [LocalDB, LocalDBNP]
    for cls in clss:
        if cls.__name__ == cfg.sqlite_cls:
            return cls(cfg.sqlite3db)
    raise ValueError("Unknown db class {!r}".format(cfg.sqlite_cls))


def sync_date_from_influx(cfg: Config):
    ts_client = connect_to_ts_database(cfg.ts_db)
    with connect_to_sqlite(cfg) as db:
        db.prepare()
        items = [(instance, None, 'virt_cpu_time') for instance in cfg.instances]
        vda_metrics = ['virt_disk_octets_write', 'virt_disk_octets_read', 'virt_disk_ops_write', 'virt_disk_ops_read']
        items += list(itertools.product(cfg.instances, ['vda'], vda_metrics))
        copy_data_from_influx(db, ts_client, items, np=True)


def parse_args(argv):
    # root passwd == masterkey
    descr = "Monitoring result analyze tool"
    parser = argparse.ArgumentParser(prog='mira-ml', description=descr)
    parser.add_argument("-l", '--log-level', default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SILENT'],
                        help="Set log level")
    parser.add_argument("-c", '--config', default='~/.config/mira-ml.yaml', help="Config file path")
    subparsers = parser.add_subparsers(dest='subparser_name')
    # ---------------------------------------------------------------------

    subparsers.add_parser('ls', help='list all vms')
    subparsers.add_parser('sync2sqlite', help='Sync data to sqlite')
    subparsers.add_parser('vm_count')
    subparsers.add_parser('lifetime')

    vm_load_corr = subparsers.add_parser('vm_load_corr')
    vm_load_corr.add_argument("-r", '--ref-vm', help="Reference vm id")
    vm_load_corr.add_argument('--start-time', help="Corr start time")
    vm_load_corr.add_argument('--stop-time', help="Corr stop time")

    node_vm_load_corr = subparsers.add_parser('node_vm_load_corr')
    node_vm_load_corr.add_argument("-r", '--ref-vm', help="Reference vm id")
    node_vm_load_corr.add_argument('--start-time', help="Corr start time")
    node_vm_load_corr.add_argument('--stop-time', help="Corr stop time")

    return parser.parse_args(argv)


# Посчитать корреляции hw-нода <-> виртуалка, найти на какой ноде живут виртуалки, сравнить с реальными данными
# Посчитать корреляции vm <-> vm, проверить как бы они агрегировались


def main(argv):
    opts = parse_args(argv[1:])
    cfg = Config()
    cfg.__dict__.update(yaml.load(open(os.path.expanduser(opts.config))))

    if opts.subparser_name == 'ls':
        client = connect_to_ts_database(cfg.ts_db)
        # list all vm in db

        vms = lookup_vms(client, cfg.instances[:32])
        vms_l = sorted(vms.values(), key=lambda x: x.lifetime)
        min_start_time = min(vm.start_time for vm in vms_l)
        max_stop_time = max(vm.stop_time for vm in vms_l)
        print(min_start_time, max_stop_time, max_stop_time - min_start_time)
        # vms_sl = vms_l[-10:]
        # fill_vms_io_stats(client, vms_sl)
        # most_loaded_vms(client)
    elif opts.subparser_name == 'sync2sqlite':
        db = connect_to_sqlite(cfg)
        db.prepare()
        copy_data_from_influx(db, cfg.ts_db, cfg.sync_metrics, cfg.instances[:32], True)

    elif opts.subparser_name == 'vm_count':
        with connect_to_sqlite(cfg) as db:
            vms = load_nodes(db, vm=True)

            vm_start_times = {vm.start_time for vm in vms}
            vm_stop_times = {vm.stop_time for vm in vms}
            all_change_times = vm_start_times.union(vm_stop_times)
            timeline = {}  # type: Dict[float, int]
            curr = 0
            for tm in sorted(all_change_times):
                curr += 1 if tm in vm_start_times else -1
                timeline[tm] = curr

            x, y = zip(*sorted(timeline.items()))
            x = [(i - x[0]) / 3600 for i in x]
            ax = pyplot.figure().add_subplot(111)
            ax.plot(x, y)
        pyplot.show()

    elif opts.subparser_name == 'lifetime':
        with connect_to_sqlite(cfg) as db:
            vms = load_nodes(db, vm=True)
            ax = pyplot.figure().add_subplot(111)
            plot_vm_lifetime(vms, ax)
        pyplot.show()

    elif opts.subparser_name == 'vm_load_corr':
        reference_vm_id = opts.ref_vm
        start_time = int(opts.start_time)
        stop_time = int(opts.stop_time)

        with connect_to_sqlite(cfg) as db:
            vms = load_nodes(db, vm=True)
            vms = [vm for vm in vms if vm.stop_time >= stop_time and vm.start_time <= start_time]

            # metr = CPUMetrics()
            # metric_slice = get_metric_slice(metr.device, metr.metric, start_time, stop_time)
            metr = disk_metric('vda', 'virt_disk_octets_write')
            metric_slice = get_metric_slice(metr.device, metr.metric, start_time, stop_time)
            dist_func = distance.euclidean

            base_vms = [vm for vm in vms if vm.name.startswith(reference_vm_id)]
            assert base_vms, "Can't find vm"
            assert len(base_vms) == 1, "Ambigious vm id"

            base_vm = base_vms[0]
            vms_l = [base_vm] + [vm for vm in vms if vm is not base_vm]
            fig = pyplot.figure(figsize=(18, 12))
            fig.set_tight_layout(True)

            gs = gridspec.GridSpec(1, 2)
            ax = pyplot.subplot(gs[0, 0])

            plot_metrics(vms_l, metric_slice, ax)
            ax.set_title("Serie values", fontsize='xx-large')

            ax = pyplot.subplot(gs[0, 1])
            vm_metric_dist_f = metric_distance(metric_slice, dist_func)
            vm_metric_dist = get_cross_vm_dist(vms_l, vm_metric_dist_f, sort=True)
            plot_cross_vm_dist(vm_metric_dist, [vm.name_short for vm in vms_l], ax)
            ax.set_title("Serie metrics", fontsize='xx-large')
            ax.title.set_position([.5, 1.1])

        pyplot.show()

    elif opts.subparser_name == 'node_vm_load_corr':
        reference_vm_id = opts.ref_vm
        start_time = int(opts.start_time)
        stop_time = int(opts.stop_time)

        with connect_to_sqlite(cfg) as db:
            hw_nodes = load_nodes(db, vm=False)
            vms = [vm for vm in load_nodes(db, vm=True)
                   if vm.stop_time >= stop_time and vm.start_time <= start_time]

            base_vms = [vm for vm in vms if vm.name.startswith(reference_vm_id)]
            assert base_vms, "Can't find vm"
            assert len(base_vms) == 1, "Ambigious vm id"
            base_vm = base_vms[0]
            vms = [base_vm] + [vm for vm in vms if vm is not base_vm]
            vms = vms[:5]

            fig = pyplot.figure(figsize=(18, 12))
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)

            dist_func = distance.correlation
            # dist_func = distance.cosine
            # dist_func = distance.euclidean

            vm_metr = disk_metric('vda', 'virt_disk_octets_write')
            vm_metric_slice = get_metric_slice(vm_metr.device, vm_metr.metric, start_time, stop_time)
            hw_metr = disk_metric('sdb', 'disk_octets_write')
            hw_metric_slice = get_metric_slice(hw_metr.device, hw_metr.metric, start_time, stop_time)

            hw_nodes.sort(key=lambda x: x.name)
            hw_vecs = list(map(hw_metric_slice, hw_nodes))
            vm_vecs = list(map(vm_metric_slice, vms))

            min_sz = min([len(i) for i in hw_vecs] + [len(i) for i in vm_vecs])
            hw_vecs = [vec[:min_sz] for vec in hw_vecs]
            hw_vecs, hw_nodes_s  = zip(*[(vec, node) for vec, node in zip(hw_vecs, hw_nodes) if vec.sum() > 1.0])

            vm_vecs = [vec[:min_sz] for vec in vm_vecs]

            # hw_vecs = [vec - vec.mean() for vec in hw_vecs]
            # vm_vecs = [vec - vec.mean() for vec in vm_vecs]

            # import IPython
            # IPython.embed()

            # vm_metric_dist = get_cross_dist(vm_vecs, hw_vecs, dist_func, sort=True)
            # plot_cross_dist(vm_metric_dist, [node.name_short for node in hw_nodes_s], [vm.name_short for vm in vms], ax)

            for vec, node in zip(hw_vecs, hw_nodes_s):
                vec = moving_average(vec, 90)
                ax.plot(vec, color='blue', label=node.name_short)

            for vec, vm in zip(vm_vecs, vms):
                vec = moving_average(vec, 90)
                ax.plot(vec, color='red', label=vm.name_short)

            ax.legend()
            ax.set_title("Serie metrics", fontsize='xx-large')
            ax.title.set_position([.5, 1.1])

        pyplot.show()
    else:
        # histo_func = metr.get_histo()
        # vms_l.remove(base_vm)
        # vms_l, dists = sort_by_hdistance(base_vm, vms_l, histo_func, dist_func)
        # vms_l.insert(0, base_vm)
        # dists.insert(0, 0)

        # fig = pyplot.figure(figsize=(36, 24))
        # reference_vm_id = '5b0f6842-a501-4c01-b78f-130530c81fa0'
        # reference_vm_id = 'e71bf537-1741-4466-acbd-6231fd083b9e'
        # reference_vm_id = 'f802ef20-f943-4d7b-890f-919b4d3edea2'
        # dist_func = distance.euclidean

        # client = InfluxDBClient('localhost', 8086, 'lma', 'lma', 'lma')
        # list_all_vms(client)
        # sync_date_from_influx()

        min_lifespan = [1.499 * 10**9, 1.49975 * 10 ** 9]

        # metr = CPUMetrics()
        # metr = IOMetricsWriteWszVda()
        metr = IOMetricsWriteVda()
        histo_func = metr.get_histo()
        # histo_func = metr2.get_histo()

        with connect_to_sqlite(cfg) as db:
        # with LocalDB("/home/koder/workspace/mira-ml/vms.db") as db:
            vms = load_vms(db)

        vms = {vm_id: vm for vm_id, vm in vms.items()
               if vm.stop_time >= min_lifespan[1] and vm.start_time <= min_lifespan[0]}

        for vm in vms.values():
            add_iop_size(vm, metr.device)

        start_time = max(vm.start_time for vm in vms.values())
        stop_time = min(vm.stop_time for vm in vms.values())

        metric_slice = get_metric_slice(metr.device, metr.metric, start_time, stop_time)
        vm_metric_dist_f = metric_distance(metric_slice, dist_func)

        # metr_f2 = get_metric_slice(CPUMetrics.device, CPUMetrics.metric, start_time, stop_time)
        # vm_metric_dist_f_cpu = metric_distance(metr_f2, dist_func)

        base_vm = vms[reference_vm_id]
        vms_l = list(vms.values())
        vms_l.remove(base_vm)
        vms_l, dists = sort_by_hdistance(base_vm, vms_l, histo_func, dist_func)
        vms_l.insert(0, base_vm)
        dists.insert(0, 0)

        # fig = pyplot.figure(figsize=(36, 24))
        fig = pyplot.figure(figsize=(18, 12))
        fig.set_tight_layout(True)

        gs = gridspec.GridSpec(1, 2)
        ax = pyplot.subplot(gs[0, 0])

        plot_metrics(vms_l, metric_slice, ax)
        ax.set_title("Serie values", fontsize='xx-large')

        ax = pyplot.subplot(gs[0, 1])
        vm_metric_dist = get_cross_vm_dist(vms_l, vm_metric_dist_f)
        plot_cross_vm_dist(vm_metric_dist, [vm.vm_id_short for vm in vms_l], ax)
        ax.set_title("Serie metrics", fontsize='xx-large')
        ax.title.set_position([.5, 1.1])

        # histo_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0, 1])
        # ax_func = lambda idx: pyplot.subplot(histo_gs[idx // 4, idx % 4])
        # plot_histos(vms_l, map(histo_func1d, vms_l), dists, ax_func)

        # dists1d = pair_distances(vms_l, histo_func1d, dist_func)
        # dists2d = pair_distances(vms_l, histo_func2d, dist_func)
        # plot_cross_vm_dist(dists1d, [vm.vm_id_short for vm in vms_l], ax)
        # ax.set_title("Histo metrics")

        # base_vm = vms_l.pop(0)
        # vms_l, dists = sort_by_hdistance(base_vm, vms_l, histo_func2d, dist_func)
        # vms_l.insert(0, base_vm)
        # dists.insert(0, 0)

        # ax_func = lambda idx: pyplot.subplot(gs[5 + idx // 4, 4 + idx % 4])
        # plot_io(vms_l, dists, ax_func, metr2.device)

        pyplot.show()
    return 0


if __name__ == "__main__":
    exit(main(sys.argv))
