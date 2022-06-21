import collections
import pathlib
from typing import Dict, Optional

import hjson


def nested_dict_update(base_dict: Dict, update_dict: Optional[Dict]):
    """
    Update a nested dictionary or similar mapping.
    Modifies d in place.
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    if update_dict is None:
        return base_dict
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            base_dict[k] = nested_dict_update(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict


def empty_callable(*args, **kwargs):
    pass


def load_params(directory: pathlib.Path):
    possible_names = ['hparams.json', 'hparams.hjson', 'params.json', 'params.hjson']
    for n in possible_names:
        filename = directory / n
        if filename.is_file():
            params = load_hjson(filename)
            return params
    raise RuntimeError(f"no params file in {directory.as_posix()}")


def load_hjson(path: pathlib.Path):
    with path.open("r") as file:
        data = hjson.load(file)
    return data
