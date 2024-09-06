from .base import AbstractDataset
from mixtime.utils import all_subclasses
from mixtime.utils import import_all_subclasses

import_all_subclasses(__file__, __name__, AbstractDataset)

DATASETS = {
    c.code(): c for c in all_subclasses(AbstractDataset) if c.code() is not None
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
