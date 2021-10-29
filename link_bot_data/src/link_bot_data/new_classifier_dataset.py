import logging
from itertools import cycle
from typing import Dict, Optional, List, Callable

import numpy as np
from more_itertools import interleave

from link_bot_data.dataset_utils import add_predicted, add_label
from link_bot_data.new_base_dataset import NewBaseDatasetLoader, NewBaseDataset
from link_bot_data.new_dataset_utils import UNUSED_COMPAT, get_filenames, load_metadata
from link_bot_data.visualization import init_viz_env, classifier_transition_viz_t, init_viz_action, \
    plot_state_t
from link_bot_pycommon.scenario_with_visualization import ScenarioWithVisualization
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from moonshine.indexing import index_time

logger = logging.getLogger(__file__)


class NewClassifierDataset(NewBaseDataset):

    def add_time(self):
        def _add_time(example: Dict):
            example['time'] = self.loader.horizon
            return example

        return _add_time

    def __init__(self,
                 loader: "NewClassifierDatasetLoader",
                 filenames: List,
                 mode: str,
                 n_prefetch=2,
                 post_process: Optional[List[Callable]] = None,
                 process_filenames: Optional[List[Callable]] = None,
                 ):
        if post_process is None:
            post_process = [self.add_time()]
        else:
            post_process.append(self.add_time())
        super().__init__(loader, filenames, mode, n_prefetch, post_process, process_filenames)

    def _only_negative(self):
        metadata = [load_metadata(f) for f in self.filenames]
        is_close = np.array([m['error'][1] < self.loader.threshold for m in metadata])
        is_far_indices, = np.where(np.logical_not(is_close))  # returns a tuple of length 1
        negative_filenames = np.take(self.filenames, is_far_indices).tolist()
        return negative_filenames

    def _balance(self, filenames):
        metadata = [load_metadata(f) for f in filenames]
        is_close = np.array([m['error'][1] < self.loader.threshold for m in metadata])
        is_close_indices, = np.where(is_close)  # returns a tuple of length 1
        is_far_indices, = np.where(np.logical_not(is_close))  # returns a tuple of length 1
        positive_filenames = np.take(filenames, is_close_indices).tolist()
        negative_filenames = np.take(filenames, is_far_indices).tolist()

        if len(filenames) == 0:
            print("Empty Dataset! balancing is meaningless")
            return filenames
        elif len(positive_filenames) == 0:
            print("No positive examples! Not balancing")
            return filenames
        elif len(negative_filenames) == 0:
            print("No negative examples! Not balancing")
            return filenames

        if len(positive_filenames) < len(negative_filenames):
            balanced_filenames = list(interleave(cycle(positive_filenames), negative_filenames))
        else:
            balanced_filenames = list(interleave(positive_filenames, cycle(negative_filenames)))
        return balanced_filenames

    def balance(self):
        def _balance_filenames(filenames):
            return self._balance(filenames)

        self._process_filenames.append(_balance_filenames)
        return self


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
        if len(filenames) == 0:
            logger.warning("No filenames, dataset will be empty.")
        dataset = NewClassifierDataset(self, filenames, mode)
        if shuffle:
            dataset = dataset.shuffle()
        if take:
            dataset = dataset.take(take)
        return dataset

    def anim_transition_rviz(self, example: Dict):
        anim = RvizAnimation(myobj=self.get_scenario(),
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

    def plot_transition(self, example: Dict, label: str, **kwargs):
        init_viz_env(self.get_scenario(), example)
        plot_state_t(self.get_scenario(), self.true_state_keys, example, t=0, label=label + '_0', **kwargs)
        plot_state_t(self.get_scenario(), self.true_state_keys, example, t=1, label=label + '_1', **kwargs)
