import numpy as np

import link_bot_pycommon.args


class LabelType(link_bot_pycommon.args.ArgsEnum):
    SDF = 0
    Overstretching = 1
    Combined = 2
    CNN_SDF = 3

    @staticmethod
    def mask(label_types):
        mask = np.zeros(len(LabelType), dtype=np.int)
        for label_type in label_types:
            i = label_type.value
            mask[i] = 1
        return mask
