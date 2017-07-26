import os
import time
from io import BytesIO, StringIO
from collections import namedtuple

import json
import yaml
import simplejson
import ujson
import marshal
import pickle
import cbor
import msgpack
import h5py
import csv

import numpy

from influx_exporter import LocalDBNP
from influxdb import InfluxDBClient

SerializerInfo = namedtuple('SerializerInfo', ['serialize_tm', 'deserialize_tm', 'size'])


def to_influx(key, x):
    pass


def to_sqlite(fname, key, x):
    db = LocalDBNP(fname)
    db.prepare()

    with db:
        db[key] = x, x

    sum_time = 0
    for i in range(100):
        tm = time.perf_counter()
        with db:
            db[key]
        sum_time += time.perf_counter() - tm

    print(int(sum_time / 100 * 1000000))


if __name__ == "__main__":

    fd_serializer_funcs = {}
    fd_deserializer_funcs = {}

    fd_serializer_funcs['json'] = ('wt', True, json.dump, json.load)
    fd_serializer_funcs['ujson'] = ('wt', True, ujson.dump, ujson.load)
    fd_serializer_funcs['simplejson'] = ('wt', True, simplejson.dump, simplejson.load)
    fd_serializer_funcs['pickle'] = ('wb', True, lambda x, fd: pickle.dump(x, fd, protocol=4), lambda fd: pickle.load(fd, protocol=4))
    fd_serializer_funcs['marchal'] = ('wb', True, marshal.dump, marshal.load)
    fd_serializer_funcs['cbor'] = ('wb', True, cbor.dump, cbor.load)
    fd_serializer_funcs['msgpack'] = ('wb', True, msgpack.dump, msgpack.load)
    fd_serializer_funcs['csv'] = ('wt', True, lambda x, fd: csv.writer(fd, delimiter=',').writerow(x), lambda fd: csv.reader(fd, delimiter=',').readrow())
    fd_serializer_funcs['numpy'] = ('wb', False, lambda x, fd: x.tofile(fd), numpy.fromfile)
    fd_serializer_funcs['numpy2'] = ('wb', False, lambda x, fd: numpy.savez(fd, vals=x), None)
    fd_serializer_funcs['numpy.save'] = ('wb', False, lambda x, fd: numpy.save(fd, x, allow_pickle=False), None)
    fd_serializer_funcs['numpy.zip'] = ('wb', False, lambda x, fd: numpy.savez_compressed(fd, vals=x), None)
    fd_serializer_funcs['numpy.txt'] = ('wb', False, lambda x, fd: numpy.savetxt(fd, x), None)
    data = numpy.cumsum(numpy.random.randn(100000))
    data_l = list(data)
    expected_test_time = 1
    select_loop_part = 0.75
    test_f_name = '/tmp/fname.bin'

    to_sqlite("/tmp/fl.bin", 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', data)
    exit(0)

    results = {}

    for name, (mode, use_py_list, serializer, deserializer) in sorted(fd_serializer_funcs.items()):
        times = []
        val = data_l if use_py_list else data

        with open(test_f_name, mode) as fd:
            estimation = time.perf_counter()
            serializer(val, fd)
            estimation = time.perf_counter() - estimation
            data_size = fd.tell()
            fd.seek(0, os.SEEK_SET)

            loops = int(expected_test_time / estimation / select_loop_part) + 1

            for i in range(loops):
                t1 = time.perf_counter()
                serializer(val, fd)
                times.append(time.perf_counter() - t1)
                fd.seek(0, os.SEEK_SET)
            times.sort()

        selected_count = int(loops * select_loop_part)
        selected_count = 1 if selected_count < 1 else selected_count
        dt = numpy.average(times[:int(selected_count)])
        results[name] = SerializerInfo(dt, 0, data_size)

    js_time, _, js_size = results.pop('json')

    frmt1 = "{:^12s}   {:^7s}     {:^8s}"
    frmt2 = "{:>12s}   {:>6.1f}x    {:>7.1f}x"
    print("Serialization/deserialization speed vs buildin json module")
    print(frmt1.format("Name", 'Speed', "Size"))
    for name, (dt, _, data_size) in sorted(results.items(), key=lambda x: x[1][0]):
        json_coef = js_time / dt
        print(frmt2.format(name, json_coef, data_size / js_size))

