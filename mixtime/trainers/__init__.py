from .base import AbstractTrainer
from mixtime.utils import all_subclasses
from mixtime.utils import import_all_subclasses

import_all_subclasses(__file__, __name__, AbstractTrainer)

TRAINERS = {
    c.code(): c for c in all_subclasses(AbstractTrainer) if c.code() is not None
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
