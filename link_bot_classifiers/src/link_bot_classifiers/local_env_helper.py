from typing import Dict

from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_point
from moonshine.moonshine_utils import add_batch, remove_batch


class LocalEnvHelper:

    def __init__(self, h: int, w: int, c: int):
        self.h = h
        self.w = w
        self.c = c
        self.indices = create_env_indices(self.h, self.w, self.c, 1)

    def get(self, center_point, environment: Dict, batch_size):
        return get_local_env_and_origin_point(center_point=center_point,
                                              environment=environment,
                                              h=self.h,
                                              w=self.w,
                                              c=self.c,
                                              indices=self.indices,
                                              batch_size=batch_size)
