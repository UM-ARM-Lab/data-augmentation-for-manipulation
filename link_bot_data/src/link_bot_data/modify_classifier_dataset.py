from collections import Callable

from link_bot_data.classifier_dataset import ClassifierDatasetLoader
from link_bot_data.load_dataset import guess_dataset_format
from link_bot_data.modify_dataset import modify_dataset, modify_dataset2
from link_bot_data.new_classifier_dataset import NewClassifierDatasetLoader
from link_bot_data.split_dataset import split_dataset


def modify_classifier_dataset(dataset_dir, suffix, process_example: Callable, save_format='pkl', hparams_update=None):
    outdir = dataset_dir.parent / f"{dataset_dir.name}+{suffix}"

    if hparams_update is None:
        hparams_update = {}

    dataset_format = guess_dataset_format(dataset_dir)
    if save_format is None:
        save_format = dataset_format

    if dataset_format == 'tfrecord':
        dataset = ClassifierDatasetLoader([dataset_dir], use_gt_rope=False, load_true_states=True)
        modify_dataset(dataset_dir=dataset_dir,
                       dataset=dataset,
                       outdir=outdir,
                       process_example=process_example,
                       save_format=save_format,
                       hparams_update=hparams_update,
                       slow=False)
    else:
        dataset = NewClassifierDatasetLoader([dataset_dir])
        modify_dataset2(dataset_dir=dataset_dir,
                        dataset=dataset,
                        outdir=outdir,
                        process_example=process_example,
                        hparams_update=hparams_update,
                        save_format=save_format)

    split_dataset(outdir)

    return outdir
