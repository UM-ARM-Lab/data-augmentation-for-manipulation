#!/usr/bin/env python
import argparse
import pathlib

import colorama
import tensorflow as tf
from colorama import Fore, Style
from google.protobuf.json_format import MessageToDict


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('--print-limit', type=int, default=25)

    args = parser.parse_args()

    if args.input.is_file():
        filenames = [args.input]
    elif not args.input.exists():
        print("directory not found")
        return
    else:
        filenames = [filename for filename in args.input.glob("*.tfrecords")]
        if len(filenames) == 0:
            print("No tfrecords found")
            return

    for filename in filenames:
        example = next(iter(tf.data.TFRecordDataset(filename.as_posix(), compression_type='ZLIB'))).numpy()
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
        print(Style.BRIGHT + filename.as_posix() + Style.NORMAL)
        if len(to_print) < args.print_limit:
            for items in to_print:
                print("{:30s}: {},".format(*items))
        else:
            for items in to_print[:args.print_limit]:
                print("{:30s}: {},".format(*items))
            if len(to_print) > 2 * args.print_limit:
                print("...")
                for items in to_print[-args.print_limit:]:
                    print("{:30s}: {},".format(*items))

        key = input(Fore.CYAN + "press enter to see an example from the next record file... (q to quit) " + Fore.RESET)
        if key == 'q':
            break


if __name__ == '__main__':
    main()
