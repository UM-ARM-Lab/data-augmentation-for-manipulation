import pathlib
from typing import Dict, Any, Optional

import hjson

from link_bot_pycommon.serialization import MyHJsonSerializer


def read_logfile(logfile_name: pathlib.Path, serializer=hjson):
    with logfile_name.open("r") as logfile:
        log = serializer.load(logfile)
    return log


def write_logfile(log: Dict, logfile_name: pathlib.Path, serializer=MyHJsonSerializer):
    with logfile_name.open("w") as logfile:
        serializer.dump(log, logfile)


class JobChunker:

    def __init__(self, logfile_name: pathlib.Path, root_log: Optional[Dict] = None, log: Optional[Dict] = None):
        self.logfile_name = logfile_name
        if root_log is not None:
            self.root_log = root_log
        else:
            if not logfile_name.exists():
                self.logfile_name.parent.mkdir(exist_ok=True, parents=True)
                self.root_log = {}
            else:
                self.root_log = read_logfile(self.logfile_name)
        if log is not None:
            self.log = log
        else:
            self.log = self.root_log

    def store_result(self, key: str, result: Any):
        self.log[key] = result
        self.save()

    def save(self):
        write_logfile(self.root_log, self.logfile_name)

    def result_exists(self, key: str):
        return key in self.log

    def setup_key(self, key: str):
        if key not in self.log:
            self.log[key] = {}
        self.save()

    def sub_chunker(self, key: str):
        sub_chunker = JobChunker(self.logfile_name, root_log=self.root_log, log=self.root_log[key])
        return sub_chunker

    def get(self, key: str):
        return self.log[key]

    def done(self, done_key='done'):
        self.log[done_key] = True
        self.save()

    def is_done(self, done_key='done'):
        return done_key in self.log and self.log[done_key]
