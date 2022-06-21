import collections
from typing import Dict, Optional


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
