import csv
from typing import Callable

from progressbar import progressbar

from link_bot_data.base_dataset import BaseDatasetLoader, SORT_FILE_NAME
from link_bot_data.progressbar_widgets import mywidgets


def sort_dataset_mode(mode: str, dataset: BaseDatasetLoader, get_value: Callable, reverse: bool):
    all_filenames = dataset.get_record_filenames(mode)
    tf_dataset = dataset.get_datasets(mode=mode, shuffle_files=False, do_not_process=True)

    values_and_record_filenames = []
    for example, record_filename in progressbar(zip(tf_dataset, all_filenames), widgets=mywidgets):
        value = get_value(dataset, example)
        values_and_record_filenames.append((value, record_filename))

    sorted_values_and_record_filenames = sorted(values_and_record_filenames, reverse=reverse)
    return sorted_values_and_record_filenames


def sort_dataset(dataset_dir, dataset: BaseDatasetLoader, get_value: Callable, reverse: bool):
    sorted_values_and_record_filenames = sort_dataset_mode('all', dataset, get_value, reverse)
    sort_filename = dataset_dir / SORT_FILE_NAME
    # https://stackoverflow.com/questions/13730107/writelines-writes-lines-without-newline-just-fills-the-file
    with open(sort_filename, 'w') as sort_file:
        writer = csv.writer(sort_file, delimiter=',')
        for value, record_filename in sorted_values_and_record_filenames:
            writer.writerow([record_filename, value, str(get_value)])
