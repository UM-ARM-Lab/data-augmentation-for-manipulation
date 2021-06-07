from typing import Dict, Tuple, Optional

import numpy as np


class BaseFilterFunction:

    def __init__(self):
        pass

    def filter(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        raise NotImplementedError()


class PassThroughFilter(BaseFilterFunction):

    def filter(self, environment: Dict, state: Optional[Dict], observation: Dict) -> Tuple[Dict, Dict]:
        mean_state = observation
        stdev_state = {k: np.zeros_like(v) for k, v in observation.items()}
        return mean_state, stdev_state
