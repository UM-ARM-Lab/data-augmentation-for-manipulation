from typing import Dict

from cylinders_simple_demo.utils import get_local_environment_torch


class LocalEnvHelper:

    def __init__(self, h: int, w: int, c: int):
        self.h = h
        self.w = w
        self.c = c
        self.indices = get_local_environment_torch.create_env_indices(self.h, self.w, self.c, 1)

    def get(self, center_point, environment: Dict, batch_size):
        return get_local_environment_torch.get_local_env_and_origin_point(center_point=center_point,
                                                                          environment=environment,
                                                                          h=self.h,
                                                                          w=self.w,
                                                                          c=self.c,
                                                                          indices=self.indices,
                                                                          batch_size=batch_size)

    def to(self, device):
        self.indices['x'] = self.indices['x'].to(device)
        self.indices['y'] = self.indices['y'].to(device)
        self.indices['z'] = self.indices['z'].to(device)
        self.indices['pixel_indices'] = self.indices['pixel_indices'].to(device)

    @property
    def device(self):
        return self.indices['x'].device
