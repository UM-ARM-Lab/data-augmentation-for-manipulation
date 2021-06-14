import pathlib
import pickle
import bsdf
import tensorflow as tf
import hjson
import numpy as np

from link_bot_pycommon.serialization import save_bsdf, load_bsdf
from moonshine.simple_profiler import SimpleProfiler


def main():
    data = {
        'error': np.array([0.0, 0.02094120718538761], np.float32),
        'z':     tf.random.uniform([3, 5]),
        'data':  'example_00000000.pkl.gz'
    }

    def _save_pkl():
        with open("/tmp/d.pkl", 'wb') as f:
            pickle.dump(data, f)

    def _load_pkl():
        with open("/tmp/d.pkl", 'rb') as f:
            data = pickle.load(f)

    def _save_hjson():
        with open("/tmp/d.hjson", 'w') as f:
            data_l = {}
            data_l['error'] = data['error'].tolist()
            data_l['data'] = data['data']
            hjson.dump(data_l, f)

    def _load_hjson():
        with open("/tmp/d.hjson", 'r') as f:
            data = hjson.load(f)
            data['error'] = np.array(data['error'])

    def _save_bsdf():
        save_bsdf(data, pathlib.Path("/tmp/d.bsdf"))

    def _load_bsdf():
        load_bsdf(pathlib.Path("/tmp/d.bsdf"))

    p = SimpleProfiler()
    n = 50000

    print('HJSON (save)', p.profile(n, _save_hjson))
    print('BSDF  (save)', p.profile(n, _save_bsdf))
    print('PKL   (save)', p.profile(n, _save_pkl))
    print('HJSON (load)', p.profile(n, _load_hjson))
    print('BSDF  (load)', p.profile(n, _load_bsdf))
    print('PKL   (load)', p.profile(n, _load_pkl))

    # data2 = {
    #     'x': 3,
    #     'y': np.random.randn(3, 3, 3),
    #     'z': tf.random.uniform([3, 5]),
    # }
    # save_bsdf(data2, pathlib.Path("/tmp/d.bsdf"))
    # data2_out = load_bsdf(pathlib.Path("/tmp/d.bsdf"))


if __name__ == '__main__':
    main()
