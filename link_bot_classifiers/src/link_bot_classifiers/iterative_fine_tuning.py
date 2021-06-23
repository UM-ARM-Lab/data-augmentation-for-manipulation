import logging
import pathlib
import time
from typing import List, Optional, Tuple

from colorama import Style, Fore

import link_bot_classifiers
from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier_from_datasets
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_pycommon.job_chunking import JobChunker
from moonshine import filepath_tools
from moonshine.filepath_tools import load_hjson
from moonshine.model_runner import ModelRunner

logger = logging.getLogger(__file__)


def chunk_dataset(dataset, chunk_sizes: List[int]):
    for s in chunk_sizes:
        yield dataset.take(s)
        dataset = dataset.skip(s)


def load_proxy_datasets(proxy_datasets_info_filename: pathlib.Path):
    proxy_datasets_info = load_hjson(proxy_datasets_info_filename)
    datasets_and_info = []
    for info in proxy_datasets_info:
        dataset_loader = get_classifier_dataset_loader([pathlib.Path(info['dataset_dir'])])
        dataset = dataset_loader.get_datasets('all')
        datasets_and_info.append((dataset_loader, dataset, info))

    return datasets_and_info


def evaluate(proxy_datasets_and_info: List[Tuple], latest_checkpoint: pathlib.Path, batch_size):
    results_list = []
    for loader, proxy_dataset, info in proxy_datasets_and_info:
        trial_path = latest_checkpoint.parent.absolute()
        _, params = filepath_tools.create_or_load_trial(trial_path=trial_path)
        model_class = link_bot_classifiers.get_model(params['model_class'])

        model = model_class(hparams=params, batch_size=batch_size, scenario=loader.get_scenario())

        runner = ModelRunner(model=model,
                             training=False,
                             params=params,
                             checkpoint=latest_checkpoint,
                             batch_metadata=loader.batch_metadata,
                             trial_path=trial_path)

        val_metrics = model.create_metrics()
        proxy_dataset = proxy_dataset.batch(batch_size).take(5)  # FIXME: this is just for debugging!
        runner.val_epoch(proxy_dataset, val_metrics)
        metric_name = info['metric']
        metric_value = val_metrics[metric_name].result().numpy().squeeze()

        status_msg = f'{[p.as_posix() for p in loader.dataset_dirs]} {metric_name} {metric_value}'
        print(Style.BRIGHT + Fore.LIGHTMAGENTA_EX + status_msg + Fore.RESET + Style.RESET_ALL)

        result = {
            metric_name:  metric_value,
            'batch_size': batch_size,
        }
        results_list.append(result)

    results = {
        'checkpoint':   latest_checkpoint.as_posix(),
        'generated-at': int(time.time()),
        'results':      results_list,
    }
    return results


def iterative_fine_tuning(training_dataset_dir: pathlib.Path,
                          checkpoint: pathlib.Path,
                          proxy_datasets_info: pathlib.Path,
                          nickname: str,
                          batch_size: int,
                          epochs: int,
                          params: Optional[pathlib.Path] = None,
                          trials_directory: pathlib.Path = pathlib.Path("./trials"),
                          augmentation_config_dir: Optional[pathlib.Path] = None):
    if params is not None:
        model_hparams_update = load_hjson(params)
    else:
        model_hparams_update = None

    dataset_loader = get_classifier_dataset_loader([training_dataset_dir])
    train_dataset = dataset_loader.get_datasets('train')
    val_dataset = dataset_loader.get_datasets('val')
    latest_checkpoint = checkpoint

    chunk_sizes = [1, 10, 100, 1000]
    train_dataset_chunks = chunk_dataset(train_dataset, chunk_sizes)
    val_dataset_chunks = chunk_dataset(val_dataset, chunk_sizes)

    proxy_datasets_and_info = load_proxy_datasets(proxy_datasets_info)

    outdir = pathlib.Path("results") / 'iterative_classifier_fine_tuning_on_datasets' / nickname

    job_chunker = JobChunker(logfile_name=outdir / 'logfile.hjson')

    for i, (train_dataset_chunk, val_dataset_chunk) in enumerate(zip(train_dataset_chunks, val_dataset_chunks)):
        job_chunker_i = job_chunker.sub_chunker(str(i))

        results_i = job_chunker_i.get_result('results')
        if results_i is None:
            results_i = evaluate(proxy_datasets_and_info, latest_checkpoint, batch_size=batch_size)
            results_i['training_dataset_dir'] = training_dataset_dir.as_posix()
            job_chunker_i.store_result('results', results_i)

        log_i = f"{nickname}/iter_{i}"
        new_trial_path = job_chunker_i.get_result('new_trial_path')
        if new_trial_path is None:
            new_trial_path = fine_tune_classifier_from_datasets(dataset_dirs=[training_dataset_dir],
                                                                batch_metadata=dataset_loader.batch_metadata,
                                                                train_dataset=train_dataset_chunk,
                                                                scenario=dataset_loader.get_scenario(),
                                                                val_dataset=val_dataset_chunk,
                                                                checkpoint=latest_checkpoint,
                                                                log=log_i,
                                                                batch_size=1,
                                                                epochs=epochs,
                                                                early_stopping=True,
                                                                fine_tune_conv=False,
                                                                fine_tune_lstm=False,
                                                                fine_tune_dense=False,
                                                                fine_tune_output=True,
                                                                model_hparams_update=model_hparams_update,
                                                                trials_directory=trials_directory,
                                                                augmentation_config_dir=augmentation_config_dir,
                                                                profile=None,
                                                                take=None)
            job_chunker_i.store_result('new_trial_path', new_trial_path)

        latest_checkpoint = new_trial_path / 'best_checkpoint'
