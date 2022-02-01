from torch.utils.data import Subset


def take_subset(dataset, take):
    if take is None:
        return dataset

    dataset_take = Subset(dataset, range(min(take, len(dataset))))
    return dataset_take