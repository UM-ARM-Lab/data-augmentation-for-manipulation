from typing import Dict

from moonshine.get_local_environment import create_env_indices, get_local_env_and_origin_point


class LocalEnvHelper:

    def __init__(self, h: int, w: int, c: int, batch_size: int):
        self.h = h
        self.w = w
        self.c = c
        self.batch_size = batch_size
        self.indices = create_env_indices(self.h, self.w, self.c, batch_size)

    def get(self, center_point, environment: Dict, batch_size):
        return get_local_env_and_origin_point(center_point=center_point,
                                              environment=environment,
                                              h=self.h,
                                              w=self.w,
                                              c=self.c,
                                              indices=self.indices,
                                              batch_size=batch_size)