import pathlib
import pickle
import tensorflow as tf
import hjson
import numpy as np

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

    p = SimpleProfiler()
    n = 50000

    print('HJSON (save)', p.profile(n, _save_hjson))
    print('PKL   (save)', p.profile(n, _save_pkl))
    print('HJSON (load)', p.profile(n, _load_hjson))
    print('PKL   (load)', p.profile(n, _load_pkl))


if __name__ == '__main__':
    main()
