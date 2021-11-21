import argparse
import pathlib

from dm_control import viewer

from link_bot_pycommon.get_scenario import get_scenario
from moonshine.filepath_tools import load_hjson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', type=pathlib.Path)
    args = parser.parse_args()

    params = load_hjson(args.params)

    s = get_scenario(params['scenario'])
    s.on_before_data_collection(params)

    viewer.launch(s.env)


if __name__ == '__main__':
    main()
