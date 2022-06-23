from torch.utils.data import Subset, ConcatDataset


def dataset_repeat(dataset, repeat: int):
    if repeat is None:
        return dataset

    dataset_repeated = ConcatDataset([dataset for _ in range(repeat)])
    return dataset_repeated


def dataset_take(dataset, take):
    if take is None:
        return dataset

    dataset_take = Subset(dataset, range(min(take, len(dataset))))
    return dataset_take


def dataset_skip(dataset, skip):
    if skip is None:
        return dataset

    dataset_take = Subset(dataset, range(skip, len(dataset)))
    return dataset_take
