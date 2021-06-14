import gzip
import json
import pathlib
import pickle
import uuid
from enum import Enum

import bsdf
import hjson
import numpy as np
import tensorflow as tf
from dataclasses_json import DataClassJsonMixin

from rospy_message_converter import message_converter
from sensor_msgs.msg import genpy


class MyHjsonEncoder(hjson.HjsonEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # if the array is of bytes we want to convert those to strings
            # this happens when you call .numpy() on a tensor of strings,
            # you get back a list bytes
            if isinstance(obj[0], bytes):
                return [b.decode("utf-8") for b in obj.tolist()]
            return obj.tolist()
        elif isinstance(obj, pathlib.Path):
            return obj.as_posix()
        elif isinstance(obj, DataClassJsonMixin):
            return obj.to_dict()
        elif np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, Enum):
            return str(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, genpy.Message):
            return message_converter.convert_ros_message_to_dictionary(obj)
        return hjson.HjsonEncoder.default(self, obj)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pathlib.Path):
            return obj.as_posix()
        elif isinstance(obj, DataClassJsonMixin):
            return obj.to_dict()
        elif np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, Enum):
            return str(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def my_dump(data, fp, indent=None):
    return json.dump(data, fp, cls=MyEncoder, indent=indent)


def my_hdump(data, fp, indent=None):
    return hjson.dump(data, fp, cls=MyHjsonEncoder)


def my_dumps(data):
    return json.dumps(data, cls=MyEncoder)


def my_hdumps(data):
    return hjson.dumps(data, cls=MyHjsonEncoder)


def dump_gzipped_pickle(data, filename):
    while True:
        try:
            with gzip.open(filename, 'wb') as data_file:
                pickle.dump(data, data_file)
            return
        except KeyboardInterrupt:
            pass


def load_gzipped_pickle(filename):
    with gzip.open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


class MyHJsonSerializer:

    @staticmethod
    def dump(data, fp):
        hjson.dump(data, fp, cls=MyHjsonEncoder)


class TfBsdfExtension(bsdf.Extension):
    name = 'tf'
    cls = (tf.Tensor, tf.Variable)

    def __init__(self):
        self.ndext = bsdf.NDArrayExtension()

    def encode(self, s, v):
        return self.ndext.encode(s, v.numpy())

    def decode(self, s, v):
        return tf.convert_to_tensor(self.ndext.decode(s, v))


my_bsdf_extensions = bsdf.standard_extensions + [TfBsdfExtension]


def load_bsdf(filename: pathlib.Path):
    with filename.open('rb') as f:
        return bsdf.load(f, extensions=my_bsdf_extensions)


def save_bsdf(data, filename: pathlib.Path):
    with filename.open('wb') as f:
        bsdf.save(f, data, extensions=my_bsdf_extensions)
