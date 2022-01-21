import pathlib
from typing import Dict, Optional, List, Callable

from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.planning_evaluation import EvaluatePlanning
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.job_chunking import JobChunker


class NoResetEvaluatePlanning(EvaluatePlanning):

    def __init__(self,
                 planner: MyPlanner,
                 service_provider: BaseServices,
                 job_chunker: JobChunker,
                 verbose: int,
                 planner_params: Dict,
                 outdir: pathlib.Path,
                 use_gt_rope=False,
                 trials: Optional[List[int]] = None,
                 record: Optional[bool] = False,
                 no_execution: Optional[bool] = False,
                 test_scenes_dir: Optional[pathlib.Path] = None,
                 extra_end_conditions: Optional[List[Callable]] = None,
                 metadata_update: Optional[Dict] = None,
                 seed: int = 0,
                 recovery_seed: int = 0,
                 ):
        super().__init__(planner,
                         service_provider=service_provider,
                         job_chunker=job_chunker,
                         verbose=verbose,
                         planner_params=planner_params,
                         outdir=outdir,
                         use_gt_rope=False,
                         trials=trials,
                         record=record,
                         no_execution=no_execution,
                         test_scenes_dir=test_scenes_dir,
                         extra_end_conditions=extra_end_conditions,
                         metadata_update=metadata_update,
                         seed=seed,
                         recovery_seed=recovery_seed)

    def on_before_run(self):
        bagfile_name = self.test_scenes_dir / f'scene_0000.bag'
        self.scenario.restore_from_bag(self.service_provider, self.planner_params, bagfile_name)

    def setup_test_scene(self, trial_idx: int):
        self.service_provider.pause()
        self.scenario.grasp_rope_endpoints(settling_time=1.0)
        self.service_provider.play()
