#!/usr/bin/env python
import argparse
import pathlib

import tensorflow as tf
from colorama import Fore, Style
from google.protobuf.json_format import MessageToDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')

    args = parser.parse_args()

    filenames = [str(filename) for filename in args.dataset_dir.glob("{}/*.tfrecords".format(args.mode))]
    for filename in filenames:
        example = next(iter(tf.data.TFRecordDataset(filename, compression_type='ZLIB'))).numpy()
        message = tf.train.Example.FromString(example)
        dict_message = MessageToDict(message)
        feature = dict_message['features']['feature']

        to_print = []
        for feature_name, feature_value in feature.items():
            type_name = list(feature_value.keys())[0]
            feature_value = feature_value[type_name]
            if 'value' in feature_value.keys():
                feature_value = feature_value['value']
                if type_name == 'bytesList':
                    to_print.append([feature_name, '<BYTES>'])
                elif type_name == 'floatList':
                    to_print.append([feature_name, len(feature_value)])
            else:
                print(Fore.RED + "Empty feature: {}, {}".format(feature_name, feature_value) + Fore.RESET)

        to_print = sorted(to_print)
        print(Style.BRIGHT + filename + Style.NORMAL)
        k = 15
        if len(to_print) < k:
            for items in to_print:
                print("{}: {},".format(*items))
        else:
            for items in to_print[:k]:
                print("{}: {},".format(*items))
            if len(to_print) > 2 * k:
                print("...")
            for items in to_print[-k:]:
                print("{}: {},".format(*items))

        key = input(Fore.CYAN + "press enter to see an example from the next record file... (q to quit) " + Fore.RESET)
        if key == 'q':
            break


if __name__ == '__main__':
    main()
