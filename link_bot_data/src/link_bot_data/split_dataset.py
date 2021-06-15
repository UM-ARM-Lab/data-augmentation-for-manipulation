from link_bot_data.dataset_utils import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT


def split_dataset(dataset_dir, extension):
    paths = sorted(list(dataset_dir.glob(f"example_*.{extension}")))
    n_files = len(paths)
    n_validation = int(DEFAULT_VAL_SPLIT * n_files)
    n_testing = int(DEFAULT_TEST_SPLIT * n_files)
    val_files = paths[0:n_validation]
    paths = paths[n_validation:]
    test_files = paths[0:n_testing]
    train_files = paths[n_testing:]

    def _write_mode(_filenames, mode):
        with (dataset_dir / f"{mode}.txt").open("w") as f:
            for _filename in _filenames:
                f.write(_filename.name + '\n')

    _write_mode(train_files, 'train')
    _write_mode(test_files, 'test')
    _write_mode(val_files, 'val')
