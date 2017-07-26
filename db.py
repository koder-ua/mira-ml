import sqlite3
from typing import Dict, Tuple, Iterable

import numpy
import msgpack

from .data import VM, AnyNumArr


class LocalDB:

    def __init__(self, path: str) -> None:
        self.db = sqlite3.connect(path)
        self.cr = None
        self.serialize = msgpack.packb
        self.deserialize = msgpack.unpackb

    def __del__(self) -> None:
        if hasattr(self, 'db'):
            self.db.close()
            self.db = None

    def prepare(self) -> None:
        cr = self.db.cursor()
        try:
            cr.execute("select key from series limit 1")
        except sqlite3.OperationalError:
            cr.execute("create table series (key text primary key, time blob, vals blob)")
            self.db.commit()

    def __enter__(self) -> 'LocalDB':
        assert self.cr is None, "Can't open new transaction, if old one is not closed"
        self.cr = self.db.cursor()
        return self

    def __exit__(self, x, y, z) -> None:
        if x is None:
            self.db.commit()
        else:
            self.db.rollback()
        self.cr = None

    def __iter__(self) -> Iterable[str]:
        self.cr.execute("select key from series")
        return (i[0] for i in self.cr.fetchall())

    def __getitem__(self, key: str) -> Tuple[AnyNumArr, AnyNumArr]:
        self.cr.execute("select time, vals from series where key=?", (key,))
        times_b, value_b = self.cr.fetchone()
        return (self.deserialize(times_b), self.deserialize(value_b))

    def __setitem__(self, key: str, val: Tuple[AnyNumArr, AnyNumArr]) -> None:
        times, values = val
        times_b = self.serialize(times)
        values_b = self.serialize(values)
        self.cr.execute("insert or replace into series values (?, ?, ?)", (key, times_b, values_b))

    def __delitem__(self, key: str) -> None:
        self.cr.execute("delete from series where key=?", (key,))


class LocalDBNP(LocalDB):
    def __init__(self, path: str, dtype=numpy.float64) -> None:
        LocalDB.__init__(self, path)
        self.serialize = lambda x: x.tobytes()
        self.deserialize = lambda x: numpy.frombuffer(x, dtype=dtype)


def load_vms(db: LocalDB) -> Dict[str, VM]:
    vms = {}

    for key in sorted(db):
        metric, instance, device = key.split(",")
        vm = vms.setdefault(instance, VM(instance))
        times, data = db[key]
        vm.metrics.setdefault(device, {})[metric] = numpy.array(data)
        vm.start_time = times[0]
        vm.stop_time = times[-1]

    return vms