from typing import Dict


def reformat_run_config_dict(config: Dict):
    out_d = {}
    for k, v in config.items():
        out_d_tmp = out_d
        sub_keys = k.split('/')
        for k_i in sub_keys[:-1]:
            if k_i not in out_d_tmp:
                out_d_tmp[k_i] = {}
            out_d_tmp = out_d_tmp[k_i]
        out_d_tmp[sub_keys[-1]] = v
    return out_d
