import logging
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tabulate
from colorama import Style, Fore

import link_bot_classifiers
from link_bot_classifiers.fine_tune_classifier import fine_tune_classifier_from_datasets
from link_bot_data.dataset_utils import compute_batch_size_for_n_examples
from link_bot_data.load_dataset import get_classifier_dataset_loader
from link_bot_pycommon import banners
from link_bot_pycommon.job_chunking import JobChunker
from link_bot_pycommon.pycommon import has_keys
from moonshine import filepath_tools
from moonshine.filepath_tools import load_hjson
from moonshine.model_runner import ModelRunner

logger = logging.getLogger(__file__)


@dataclass
class ResultMetadata:
    seed: int
    augmentation_type: str
    training_dataset_dir: str
    train_dataset_chunk_len: int
    val_dataset_chunk_len: int


def chunk_dataset(dataset, chunk_sizes: List[int]):
    datasets = []
    for take in np.cumsum(chunk_sizes):
        datasets.append(dataset.take(take))
    return datasets


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
        # proxy_dataset = proxy_dataset.batch(batch_size)
        proxy_dataset = proxy_dataset.take(241).batch(batch_size)  # FIXME: this is just for debugging!
        runner.val_epoch(proxy_dataset, val_metrics)
        metric_name = info['metric']
        metric_value = val_metrics[metric_name].result().numpy().squeeze()

        result = {
            'metric_name':  metric_name,
            'metric_value': metric_value,
            'info':         info,
            'batch_size':   batch_size,
        }
        results_list.append(result)

    results = {
        'checkpoint':   latest_checkpoint.as_posix(),
        'generated-at': int(time.time()),
        'results':      results_list,
    }
    return results


def check_and_fine_tune(job_chunker_i, dataset_loader, training_dataset_dir, train_chunk, val_chunk, **kwargs):
    new_trial_path = job_chunker_i.get_result('new_trial_path')
    if new_trial_path is None:
        print(banners.equals("FINE-TUNE"))
        adaptive_batch_size = compute_batch_size_for_n_examples(total_examples=len(train_chunk),
                                                                max_batch_size=32)
        new_trial_path = fine_tune_classifier_from_datasets(dataset_dirs=[training_dataset_dir],
                                                            batch_metadata=dataset_loader.batch_metadata,
                                                            train_dataset=train_chunk,
                                                            scenario=dataset_loader.get_scenario(),
                                                            val_dataset=val_chunk,
                                                            batch_size=adaptive_batch_size,
                                                            **kwargs)
        job_chunker_i.store_result('new_trial_path', new_trial_path)
    return new_trial_path


def check_and_evaluate(job_chunker_i, batch_size, latest_checkpoint, proxy_datasets_and_info, m_i: ResultMetadata):
    results_i = job_chunker_i.get_result('results')
    if results_i is None:
        print(banners.equals("EVALUATE"))
        results_i = evaluate(proxy_datasets_and_info, latest_checkpoint, batch_size=batch_size)
        results_i['seed'] = m_i.seed
        results_i['augmentation_type'] = m_i.augmentation_type
        results_i['training_dataset_dir'] = m_i.training_dataset_dir
        results_i['train_dataset_size'] = m_i.train_dataset_chunk_len
        results_i['val_dataset_size'] = m_i.val_dataset_chunk_len
        job_chunker_i.store_result('results', results_i)
    return results_i


def print_results(iter_idx, results_i):
    headers = ['iter', 'dataset', 'metric_name', 'metric_value']
    table = []
    for r in results_i['results']:
        row = [
            iter_idx,
            r['metric_name'],
            r['metric_value'] * 100,
        ]
        table.append(row)

    table_str = tabulate.tabulate(table,
                                  headers=headers,
                                  tablefmt=tabulate.simple_separated_format("\t"),
                                  numalign='left')
    print('=' * 80)
    print(Style.BRIGHT + Fore.LIGHTMAGENTA_EX + table_str + Fore.RESET + Style.RESET_ALL)


def iterative_fine_tuning(training_dataset_dir: pathlib.Path,
                          checkpoint: pathlib.Path,
                          proxy_datasets_info: pathlib.Path,
                          nickname: str,
                          batch_size: int,
                          epochs: int,
                          seed: int,
                          params: Optional[pathlib.Path] = None,
                          augmentation_config_dir: Optional[pathlib.Path] = None,
                          **kwargs):
    if params is not None:
        model_hparams_update = load_hjson(params)
    else:
        model_hparams_update = {}

    print(Fore.YELLOW + f'{seed=}' + Fore.RESET)

    dataset_loader = get_classifier_dataset_loader([training_dataset_dir])
    train_dataset = dataset_loader.get_datasets('train', shuffle=seed)
    val_dataset = dataset_loader.get_datasets('val', shuffle=seed)
    latest_checkpoint = checkpoint

    chunk_sizes = [1, 10, 100, 1000]
    train_dataset_chunks = chunk_dataset(train_dataset, chunk_sizes)
    val_dataset_chunks = chunk_dataset(val_dataset, chunk_sizes)

    proxy_datasets_and_info = load_proxy_datasets(proxy_datasets_info)

    outdir = pathlib.Path("results") / 'iterative_classifier_fine_tuning_on_datasets' / nickname / f'seed_{seed}'

    job_chunker = JobChunker(logfile_name=outdir / 'logfile.hjson')

    m_i = None
    i = 0
    for i, (train_chunk, val_chunk) in enumerate(zip(train_dataset_chunks, val_dataset_chunks)):
        m_i = ResultMetadata(seed=seed,
                             augmentation_type=has_keys(model_hparams_update, ['augmentation', 'type'], noop_val=None),
                             training_dataset_dir=training_dataset_dir.as_posix(),
                             train_dataset_chunk_len=len(train_chunk),
                             val_dataset_chunk_len=len(val_chunk))

        job_chunker_i = job_chunker.sub_chunker(str(i))

        results_i = check_and_evaluate(job_chunker_i, batch_size, latest_checkpoint, proxy_datasets_and_info, m_i)
        print_results(iter_idx=i, results_i=results_i)

        new_trial_path = check_and_fine_tune(job_chunker_i,
                                             dataset_loader,
                                             training_dataset_dir,
                                             train_chunk,
                                             val_chunk,
                                             log=f"{nickname}/iter_{i}",
                                             checkpoint=latest_checkpoint,
                                             epochs=epochs,
                                             augmentation_config_dir=augmentation_config_dir)

        latest_checkpoint = pathlib.Path(new_trial_path) / 'best_checkpoint'

    # One final evaluate
    i += 1
    job_chunker_i = job_chunker.sub_chunker(str(i))
    results_i = check_and_evaluate(job_chunker_i, batch_size, latest_checkpoint, proxy_datasets_and_info, m_i)
    print_results(iter_idx=i, results_i=results_i)
