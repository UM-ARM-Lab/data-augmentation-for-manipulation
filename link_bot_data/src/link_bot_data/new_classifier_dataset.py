import pathlib
from itertools import cycle
from typing import Dict, Optional, List, Callable

import halo
import hjson
import numpy as np
from more_itertools import interleave

from link_bot_data.dataset_utils import add_predicted, add_label
from link_bot_data.new_base_dataset import NewBaseDatasetLoader, NewBaseDataset
from link_bot_data.new_dataset_utils import UNUSED_COMPAT, get_filenames, load_metadata
from link_bot_data.visualization import init_viz_env, init_viz_action, classifier_transition_viz_t
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.filepath_tools import load_hjson
from moonshine.indexing import index_time


class NewClassifierDataset(NewBaseDataset):

    def add_time(self):
        def _add_time(example: Dict):
            example['time'] = self.loader.horizon
            return example

        return _add_time

    def __init__(self, loader: "NewClassifierDatasetLoader", filenames: List, mode: str,
                 post_process: Optional[List[Callable]] = None, n_prefetch=2):
        if post_process is None:
            post_process = [self.add_time()]
        else:
            post_process.append(self.add_time())
        super().__init__(loader, filenames, mode, post_process, n_prefetch)

    @halo.Halo("balancing")
    def _balance(self):
        metadata = [load_metadata(f) for f in self.filenames]
        is_close = np.array([m['error'][1] < self.loader.threshold for m in metadata])
        is_close_indices, = np.where(is_close)  # returns a tuple of length 1
        is_far_indices, = np.where(np.logical_not(is_close))  # returns a tuple of length 1
        positive_filenames = np.take(self.filenames, is_close_indices).tolist()
        negative_filenames = np.take(self.filenames, is_far_indices).tolist()
        if len(positive_filenames) == 0:
            print("No positive examples! Not balancing")
            return self.filenames
        if len(negative_filenames) == 0:
            print("No negative examples! Not balancing")
            return self.filenames
        if len(positive_filenames) < len(negative_filenames):
            balanced_filenames = list(interleave(cycle(positive_filenames), negative_filenames))
        else:
            balanced_filenames = list(interleave(positive_filenames, cycle(negative_filenames)))
        return balanced_filenames

    def balance(self):
        all_balanced_filenames = []
        for root in self.loader.dataset_dirs:
            balance_filename = root / 'balanced.hjson'
            if balance_filename.exists():
                balance_info = load_hjson(balance_filename)
            else:
                balance_info = {}

            if self.mode in balance_info and str(self.loader.threshold) in balance_info[self.mode]:
                balanced_filenames = [pathlib.Path(f) for f in balance_info[self.mode][str(self.loader.threshold)]]
            else:
                balanced_filenames = self._balance()
                balance_info[self.mode] = {}
                balance_info[self.mode][str(self.loader.threshold)] = [f.as_posix() for f in balanced_filenames]
                with balance_filename.open("w") as bf:
                    hjson.dump(balance_info, bf)

            all_balanced_filenames.extend(balanced_filenames)

        return NewClassifierDataset(self.loader, balanced_filenames, self.mode, self._post_process, self.n_prefetch)


class NewClassifierDatasetLoader(NewBaseDatasetLoader):

    def __init__(self, dataset_dirs,
                 threshold: Optional[float] = None,
                 scenario: Optional[ScenarioWithVisualization] = None,
                 load_true_states=UNUSED_COMPAT,
                 old_compat=UNUSED_COMPAT,
                 use_gt_rope=UNUSED_COMPAT,
                 verbose=UNUSED_COMPAT):
        super().__init__(dataset_dirs, scenario)

        self.labeling_params = self.hparams['labeling_params']
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        self.true_state_keys = self.hparams['true_state_keys']
        self.state_metadata_keys = self.hparams['state_metadata_keys']
        self.predicted_state_keys = [add_predicted(k) for k in self.hparams['predicted_state_keys']]
        self.threshold = threshold if threshold is not None else self.labeling_params['threshold']
        self.predicted_state_keys.append(add_predicted('stdev'))
        self.env_keys = self.hparams['env_keys']
        self.action_keys = self.hparams['action_keys']

    def post_process(self, e):
        add_label(e, self.threshold)
        return e

    def get_datasets(self,
                     mode: str,
                     shuffle: bool = False,
                     take: int = None,
                     do_not_process=UNUSED_COMPAT,
                     slow=UNUSED_COMPAT):
        filenames = get_filenames(self.dataset_dirs, mode)
        assert len(filenames) > 0
        dataset = NewClassifierDataset(self, filenames, mode)
        if shuffle:
            dataset = dataset.shuffle()
        if take:
            dataset = dataset.take(take)
        return dataset

    def anim_transition_rviz(self, example: Dict):
        anim = RvizAnimation(scenario=self.get_scenario(),
                             n_time_steps=self.horizon,
                             init_funcs=[init_viz_env, self.init_viz_action()],
                             t_funcs=[init_viz_env, self.classifier_transition_viz_t()])
        anim.play(example)

    def index_true_state_time(self, example: Dict, t: int):
        return index_time(example, self.true_state_keys, t, False)

    def index_pred_state_time(self, example: Dict, t: int):
        return index_time(example, self.predicted_state_keys, t, False)

    def classifier_transition_viz_t(self):
        return classifier_transition_viz_t(metadata={},
                                           state_metadata_keys=self.state_metadata_keys,
                                           predicted_state_keys=self.predicted_state_keys,
                                           true_state_keys=self.true_state_keys)

    def init_viz_action(self):
        return init_viz_action({}, self.action_keys, self.predicted_state_keys)
