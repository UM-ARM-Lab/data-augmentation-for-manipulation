from typing import Dict


class LocalEnvHelper:

    def __init__(self, h: int, w: int, c: int, get_local_env_module=None):
        if get_local_env_module is None:
            from moonshine import get_local_environment_tf
            self.get_local_env_module = get_local_environment_tf
        else:
            self.get_local_env_module = get_local_env_module
        self.h = h
        self.w = w
        self.c = c
        self.indices = self.get_local_env_module.create_env_indices(self.h, self.w, self.c, 1)

    def get(self, center_point, environment: Dict, batch_size):
        return self.get_local_env_module.get_local_env_and_origin_point(center_point=center_point,
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
