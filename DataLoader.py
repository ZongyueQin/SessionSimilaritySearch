from collections.abc import Mapping, Sequence
from inspect import signature
from typing import List, Optional, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)

class MyCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.collater = Collater(follow_batch, exclude_keys)
    
    def __call__(self, batch):
        l = len(batch[0])
        ret = []
        for i in range(l):
            batch_data = [data[i] for data in batch]
            ret.append(self.collater(batch_data))
#        batch_0 = [data[0] for data in batch]
#        batch_1 = [data[1] for data in batch]
        return tuple(ret)

# PyG 'Data' objects are subclasses of MutableMapping, which is an
# instance of collections.abc.Mapping. Currently, PyTorch pin_memory
# for DataLoaders treats the returned batches as Mapping objects and
# calls `pin_memory` on each element in `Data.__dict__`, which is not
# desired behavior if 'Data' has a `pin_memory` function. We patch
# this behavior here by monkeypatching `pin_memory`, but can hopefully patch
# this in PyTorch in the future:
__torch_pin_memory = torch.utils.data._utils.pin_memory.pin_memory
__torch_pin_memory_params = signature(__torch_pin_memory).parameters


def pin_memory(data, device=None):
    if hasattr(data, "pin_memory"):
        return data.pin_memory()
    if len(__torch_pin_memory_params) > 1:
        return __torch_pin_memory(data, device)
    return __torch_pin_memory(data)


torch.utils.data._utils.pin_memory.pin_memory = pin_memory


class MyDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=MyCollater(follow_batch, exclude_keys),
            **kwargs,
        )
