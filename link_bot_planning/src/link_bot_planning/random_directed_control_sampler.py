import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.control as oc


class RandomDirectedControlSampler(oc.DirectedControlSampler):

    def __init__(self, si, seed: int, my_planner):
        super().__init__(si)
        self.si = si
        self.my_planner = my_planner
        self.control_space = self.si.getControlSpace()
        self.control_sampler = self.control_space.allocControlSampler()
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @classmethod
    def allocator(cls, seed, my_planner):
        def alloc(si):
            return cls(si, seed, my_planner)

        return oc.DirectedControlSamplerAllocator(alloc)

    def sampleTo(self, control_out, previous_control, state, target_out):
        # how do we do this?
        min_step_count = self.si.getMinControlDuration()
        max_step_count = self.si.getMaxControlDuration()
        self.control_sampler.sample(control_out)

        return np.uint32(self.rng.randint(min_step_count, max_step_count))
