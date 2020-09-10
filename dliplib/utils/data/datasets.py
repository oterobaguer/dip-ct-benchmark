import numpy as np

from dival import Dataset
from dliplib.utils.helper import ReorderedDataset


class CachedDataset(Dataset):
    """Dataset that allows to use cached elements of a dataset
    """

    def __init__(self, dataset, space, cache_files, size_part=1.0):
        """
        Parameters
        ----------
        dataset : :class:`.Dataset`
            Original CT dataset from which the ground truth is used.
        space : [tuple of ] :class:`odl.space.base_tensors.TensorSpace`,\
                optional
            The spaces of the elements of samples as a tuple.
            It is strongly recommended to set this parameter since the spaces of the
            resulting dataset might be different from the original one. E.g. the original
            dataset may contain pairs of (obs, gt) and the new dataset might contain (fbp, gt)
        cache_files : dict
            Filenames of the cache files for each file and for each component
            The part (``'train'``, ...) is the key to the dict, for each part two filenames
            should be provided, each of them can be None: meaning that this component should
            be fetched from the original CT dataset.
            To generate the FBP files, :func:`generate_fbp_cache` can be used.
        """
        super().__init__(space=space)

        self.cache_files = cache_files
        self.data = {}

        for part in ['train', 'validation']:
            self.data[part] = []
            for k in range(2):
                if cache_files[part][k]:
                    try:
                        self.data[part].append(
                            np.load(cache_files[part][k], mmap_mode='r'))
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            "Did not find cache file '{}'".format(cache_files[part][k]))
                else:
                    self.data[part].append(None)

        self.dataset = dataset
        self.reorder_idx = (self.dataset.idx if isinstance(self.dataset, ReorderedDataset)
                            else None)
        self.train_len = max(1, int(size_part * self.dataset.train_len))
        self.validation_len = max(
            1, int(size_part * self.dataset.validation_len))

        self.random_access = True

    def get_sample(self, index, part='train', out=None):
        # TODO: use out for a more efficient use of memory
        if index >= self.get_len(part):
            raise IndexError(
                "index {:d} out of range for dataset part '{}' (len: {:d})"
                .format(index, part, self.get_len(part)))
        data_idx = index if self.reorder_idx is None else self.reorder_idx[part][index]
        if self.data[part][0] is None:
            first = self.dataset.get_sample(index,
                                            part=part,
                                            out=(True, False))[0]
        else:
            first = self.data[part][0][data_idx]
        if self.data[part][1] is None:
            second = self.dataset.get_sample(index,
                                             part=part,
                                             out=(False, True))[1]
        else:
            second = self.data[part][1][data_idx]
        return first, second
