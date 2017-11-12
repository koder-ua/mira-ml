import re
import time
import queue
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Tuple, List, Optional, Any, cast, Dict, Union

import numpy
from influxdb import InfluxDBClient

from cephlib.units import b2ssize_10, b2ssize

from .db import LocalDB
from .data import influxtime2tm, Node, TSStatInfo, parse_db_creds, TSDBCreds


def parse(descr: str) -> Tuple[str, Dict[str, str]]:
    name, *params = descr.split(",")
    attrs = dict(param.split('=') for param in params)
    return name, attrs


# find all vms
def list_all_vm_io(conn: Any) -> None:
    client = cast(InfluxDBClient, conn)
    series = sorted(list(client.query('SHOW SERIES from "virt_disk_ops_read"')))
    hd_series = {}
    for serie in series[0]:
        name, attrs = parse(serie['key'])
        hd_series.setdefault(name, []).append(attrs)

    print({serie['device'] for serie in hd_series['virt_disk_ops_read']})
    vm_io = {}

    for idx, serie in enumerate(hd_series['virt_disk_ops_read']):
        if serie['device'] == 'vda':
            q = "select value from virt_disk_ops_read where instance_id='{instance_id}' and device='{device}'".format(**serie)
            for io_serie in client.query(q):
                vm_io[serie['instance_id']] = sum(i['value'] for i in io_serie)
                # print(serie['instance_id'], sum(i['value'] for i in io_serie))
        if idx % 100 == 0:
            print(100 * idx / len(hd_series['virt_disk_ops_read']), "% done")

    for io_sz, iid in list(sorted([(-sz, vid) for vid, sz in vm_io.items()]))[:50]:
        print(iid, int(-io_sz))


IO_METRICS = [
    ('virt_disk_ops_read', 'device', 'ops'),
    ('virt_disk_ops_write', 'device', 'ops'),
    ('virt_disk_octets_read', 'device', 'B'),
    ('virt_disk_octets_write', 'device', 'B'),
    ('virt_if_octets_rx', 'interface', 'B'),
    ('virt_if_octets_tx', 'interface', 'B'),
    ('virt_if_packets_rx', 'interface', 'pkt'),
    ('virt_if_packets_tx', 'interface', 'pkt'),
]


def fill_vms_io_stats(client: InfluxDBClient, vms: List[Node]) -> None:
    metr_stat_t = "select sum(value),mean(value),median(value),stddev(value) from {} where instance_id='{}' group by {}"
    for vm in vms:
        for metr, dev, units in IO_METRICS:
            metric_q = metr_stat_t.format(metr, vm.vm_id, dev)
            for (_, tags), data in client.query(metric_q).items():
                dt = list(data)[0]
                sinfo = vm.metrics.setdefault(metr, {}).setdefault(tags[dev], TSStatInfo())
                sinfo.mean = dt['mean']
                sinfo.median = dt['median']
                sinfo.stddev = dt['stddev']
                sinfo.total = dt['sum']
                sinfo.units = units


def get_vms_io_stats(client: InfluxDBClient) -> Dict[str, Tuple[float, float, float]]:
    metr_stat_r = "select sum(value) from virt_disk_octets_read group by instance_id"
    metr_stat_w = "select sum(value) from virt_disk_octets_write group by instance_id"
    disk_io = {}
    for q in (metr_stat_r, metr_stat_w):
        for (metric, tags), serie in client.query(q).items():
            vm_id = tags['instance_id']
            disk_io[vm_id] = disk_io.get(vm_id, 0) + list(serie)[0]['sum']

    metr_stat_r = "select sum(value) from virt_if_octets_rx group by instance_id"
    metr_stat_w = "select sum(value) from virt_if_octets_tx group by instance_id"
    net_io = {}
    for q in (metr_stat_r, metr_stat_w):
        for (metric, tags), serie in client.query(q).items():
            vm_id = tags['instance_id']
            net_io[vm_id] = net_io.get(vm_id, 0) + list(serie)[0]['sum']

    metr_stat_r = "select sum(value) from virt_cpu_time group by instance_id"
    cpu_usage = {}
    for (metric, tags), serie in client.query(metr_stat_r).items():
        vm_id = tags['instance_id']
        cpu_usage[vm_id] = list(serie)[0]['sum']

    return {instance_id: (disk_io[instance_id], net_io[instance_id], cpu_usage[instance_id])
            for instance_id in disk_io}


def most_loaded_vms(conn: Any, count: int = 100) -> List[str]:
    client = cast(InfluxDBClient, conn)
    io_stats = get_vms_io_stats(client)

    by_cpu = [instance_id for instance_id, _ in sorted(io_stats.items(), key=lambda x: -x[1][2])]
    by_hdd = [instance_id for instance_id, _ in sorted(io_stats.items(), key=lambda x: -x[1][0])]
    by_net = [instance_id for instance_id, _ in sorted(io_stats.items(), key=lambda x: -x[1][1])]

    ids = by_cpu[count:] + by_hdd[count:] + by_net[count:]
    sorted_by_count = sorted(set(ids), key=lambda x: by_cpu.index(x) + by_hdd.index(x) + by_net.index(x))

    for vm_id in sorted_by_count[:count]:
        print("id={} cpu={}s disk={}B net={}B".format(vm_id,
                                                      b2ssize_10(int(io_stats[vm_id][2] / 1E9)),
                                                      b2ssize(io_stats[vm_id][0]),
                                                      b2ssize(io_stats[vm_id][1])))

    return []


def lookup_vms(conn: Any, vm_ids: List[str]) -> Dict[str, Node]:
    client = cast(InfluxDBClient, conn)

    q = "select first(value) from virt_cpu_time group by instance_id"
    vms = {}  # type: Dict[str, Node]
    for (metric, tags), serie in client.query(q).items():
        vm_id = tags['instance_id']
        if vm_id in vm_ids:
            vm = vms[vm_id] = Node(vm_id, is_vm=True)
            vm.start_time = influxtime2tm(list(serie)[0]['time'])

    q = "select last(value) from virt_cpu_time group by instance_id"
    for (metric, tags), serie in client.query(q).items():
        vm_id = tags['instance_id']
        if vm_id in vm_ids:
            vms[vm_id].stop_time = influxtime2tm(list(serie)[0]['time'])

    return vms


def list_hw_hosts(client: InfluxDBClient) -> List[str]:
    res = client.query("select first(value) from contextswitch group by hostname")
    return [params['hostname'] for (_, params), _ in res.items()]


def list_host_devs_for_metric(client: InfluxDBClient, hostname: str, metric: str, dev_tp: str) -> List[str]:
    res = client.query("select first(value) from {} where hostname='{}' group by {}".format(metric, hostname, dev_tp))
    return [params[dev_tp] for (_, params), _ in res.items()]


def list_vm_devs_for_metric(client: InfluxDBClient, vm_id: str, metric: str, dev_tp: str) -> List[str]:
    res = client.query("select first(value) from {} where instance_id='{}' group by {}".format(metric, vm_id, dev_tp))
    return [params[dev_tp] for (_, params), _ in res.items()]


def data_loader(node: str,
                metric_name: str,
                dev: Optional[str],
                dev_tp: Optional[str],
                is_vm: bool,
                min_time: Optional[int],
                max_time: Optional[int],
                conn_pool: queue.Queue) -> Tuple[str, str, str, bool, List[List]]:
    try:
        client = conn_pool.get()
        q = "select value from " + metric_name

        if is_vm:
            q += " where instance_id='{}'".format(node)
        else:
            q += " where hostname='{}'".format(node)

        if dev:
            q += " and {}='{}'".format(dev_tp, dev)

        if min_time:
            q += " and time >= {}s".format(min_time)

        if max_time:
            q += " and time <= {}s".format(max_time)

        print(q)

        series = client.query(q).items()
        conn_pool.put(client)
        return node, metric_name, str(dev), is_vm, series
    except Exception as exc:
        print(exc)
        raise


# https://github.com/influxdata/influxdb/issues/139
def copy_data_from_influx(db: LocalDB,
                          conn_uri: str,
                          items: Dict[str, Union[str, Dict[str, str]]],
                          instances: List[str],
                          np: bool = False,
                          conn_count: int = 2,
                          replace: bool = False):

    conn_pool = queue.Queue()
    for _ in range(conn_count):
        client = connect_to_ts_database(conn_uri)
        conn_pool.put(client)

    hw_hosts = list_hw_hosts(client)

    if 'hw_node_filter' in items:
        hw_filter = re.compile(items['hw_node_filter'] + "$")
        hw_hosts = list(filter(hw_filter.match, hw_hosts))

    if 'time_limit' in items:
        min_time, max_time = items['time_limit']
    else:
        min_time = max_time = None

    units_dct = {}  # type: Dict[str, str]

    with db:
        db_keys = set(db)

    all_metrics_to_sync = []
    for node_type, series in items.items():

        if node_type == 'vm':
            is_vm = True
        elif node_type == 'hw_node':
            is_vm = False
        else:
            continue

        for metric, params in series.items():
            if ',' in params:
                group_by, units, name_re = params.split(",", 2)
            else:
                group_by, units, name_re = None, params, None

            assert units_dct.get(metric) is None or units_dct[metric] == units
            units_dct[metric] = units

            if is_vm:
                func = list_vm_devs_for_metric
                lst = instances
            else:
                func = list_host_devs_for_metric
                lst = hw_hosts

            for host in lst:
                if group_by is not None:
                    devs = func(client, host, metric, group_by)
                else:
                    devs = [None]

                for dev in devs:
                    if name_re is not None:
                        if not re.match(name_re + "$", dev):
                            continue
                    key = "{},{},{},{}".format(metric, host, dev, '1' if is_vm else '0')
                    if key not in db_keys:
                        all_metrics_to_sync.append((host, metric, dev, group_by, is_vm, min_time, max_time, conn_pool))

    all_metrics_to_sync.sort()
    total_ts = len(all_metrics_to_sync)
    with ThreadPoolExecutor(max_workers=conn_count) as executor:
        influxtime2tm_l = influxtime2tm
        toutc = time.mktime(time.localtime()) - time.mktime(time.gmtime())

        futures = [executor.submit(data_loader, *mtr) for mtr in all_metrics_to_sync[:conn_count * 2]]
        all_metrics_to_sync = all_metrics_to_sync[conn_count * 2:]

        while futures or all_metrics_to_sync:
            node, metric, device, is_vm, series = futures.pop().result()
            key = "{},{},{},{}".format(metric, node, device, '1' if is_vm else '0')

            if all_metrics_to_sync:
                futures.append(executor.submit(data_loader, *all_metrics_to_sync.pop()))

            if len(series) == 0:
                print("No date for key", key)
                continue

            assert len(series) == 1

            (_, group_by_dct), data_gen = list(series)[0]

            t = time.time()
            assert group_by_dct is None
            hostname = node[:8] if is_vm else node

            data_gen = list(data_gen)
            iter1 = (influxtime2tm_l(i['time']) for i in data_gen)
            iter2 = (i['value'] for i in data_gen)

            if np:
                times = numpy.fromiter(iter1, dtype=float) - toutc
                values = numpy.fromiter(iter2, dtype=float)
            else:
                times = list(i - toutc for i in iter1)
                values = list(iter2)

            with db:
                db[key] = (times, values)

            perc_left = (len(futures) + len(all_metrics_to_sync) * 100)// total_ts
            msg_t = "{host:>10s} {dev:>15s} {metr:>25s} {time:>3.1f}s  size = {size:>3d}k {perc:>2d}% left to sync"
            msg = msg_t.format(host=hostname, dev=str(device), metr=metric, time=time.time() - t,
                               size=len(times) // 1024, perc=perc_left)
            print(msg)


def connect_to_ts_database(uri: str) -> Any:
    proto, uri = uri.split("://", 1)
    creds = parse_db_creds(uri)
    if proto == 'influx':
        default_port = 8086
        port = creds.port if creds.port is not None else default_port
        return InfluxDBClient(creds.host, port, creds.user, creds.password, creds.database)

    raise ValueError("Unknown proto: {}".format(proto))
