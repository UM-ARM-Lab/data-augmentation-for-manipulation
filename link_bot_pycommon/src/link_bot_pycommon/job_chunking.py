import json
import pathlib
import pickle
import shutil
from typing import Dict, Any, Optional

import hjson

from link_bot_pycommon.serialization import MyHJsonSerializer


def guess_mode_for_serializer(mode, serializer):
    if serializer in [hjson, json]:
        return mode
    if serializer in [pickle]:
        return mode + 'b'


def read_logfile(logfile_name: pathlib.Path, serializer=hjson):
    with logfile_name.open(guess_mode_for_serializer('r', serializer)) as logfile:
        log = serializer.load(logfile)
    return log


def write_logfile(log: Dict, logfile_name: pathlib.Path, serializer=MyHJsonSerializer):
    # prevents the user from losing the logfile by interrupting and killing the program
    temp_logfile_name = logfile_name.parent / (logfile_name.name + ".tmp")
    with temp_logfile_name.open(guess_mode_for_serializer('w', serializer)) as logfile:
        serializer.dump(log, logfile)
    shutil.copy(temp_logfile_name, logfile_name)


class JobChunker:

    def __init__(self,
                 logfile_name: pathlib.Path,
                 root_log: Optional[Dict] = None,
                 log: Optional[Dict] = None,
                 serializer=hjson):
        self.logfile_name = logfile_name
        self.serializer = serializer
        if root_log is not None:
            self.root_log = root_log
        else:
            if not logfile_name.exists():
                self.logfile_name.parent.mkdir(exist_ok=True, parents=True)
                self.root_log = {}
            else:
                self.root_log = read_logfile(self.logfile_name, serializer=self.serializer)
        if log is not None:
            self.log = log
        else:
            self.log = self.root_log

    def store_result(self, key: str, result: Any, save=True):
        self.log[key] = result
        if save:
            self.save()

    def store_results(self, update_dict: Dict, save=True):
        self.log.update(update_dict)
        if save:
            self.save()

    def save(self):
        write_logfile(self.root_log, self.logfile_name, serializer=self.serializer)

    def get_result(self, key: str):
        return self.log.get(key, None)

    def has_result(self, key: str):
        return key in self.log

    def setup_key(self, key: str):
        if key not in self.log:
            self.log[key] = {}
        self.save()

    def sub_chunker(self, key: str):
        self.setup_key(key)
        sub_chunker = JobChunker(self.logfile_name, root_log=self.root_log, log=self.log[key])
        return sub_chunker

    def get(self, key: str):
        return self.log[key]

    def load_prompt(self, key, *args):
        """
        Loads the value of key from the logfile, or prompts the user if it's not found.
        If you provide a second argument, it will be used as a default.
        Default is returned if key is not in the logfile, and the user enters nothing (simply presses enter)
        Args:
            key: string
            *args: an optional default arg

        Returns:
            the value

        """
        has_default = False
        default = None
        if len(args) == 1:
            default = args[0]
            has_default = True

        if key in self.log:
            return self.log[key]

        if has_default:
            v = input(f"{key} [{default}]: ")
            if v == '':
                v = default
        else:
            v = input(f"{key}: ")

        self.store_result(key, v)
        return v
