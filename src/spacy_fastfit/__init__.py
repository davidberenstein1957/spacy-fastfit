import logging
from typing import TYPE_CHECKING, Union

# from rich.logging import RichHandler
from spacy.language import Language

from spacy_fastfit.models import SpacyFastFit
from spacy_fastfit.schemas import FastFitTrainerArgs

if TYPE_CHECKING:
    from datasets import Dataset


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
)


@Language.factory(
    "spacy_fastfit",
    default_config={
        "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "train_dataset": None,
        "validation_dataset": None,
        "num_train_epochs": 40,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "max_text_length": 512,
        "dataloader_drop_last": False,
        "num_repeats": 4,
        "optim": "adafactor",
        "clf_loss_factor": 0.1,
        "fp16": True,
        "multi_label": False,
        "seed": None,
    },
)
def create_fastfit_model(
    nlp: Language,
    name: str,
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    train_dataset: Union[dict, "Dataset", None] = None,
    validation_dataset: Union[dict, "Dataset", None] = None,
    num_train_epochs: int = 40,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    max_text_length: int = 512,
    dataloader_drop_last: bool = False,
    num_repeats: int = 4,
    optim: str = "adafactor",
    clf_loss_factor: float = 0.1,
    fp16: bool = True,
    multi_label: bool = False,
    seed: Union[int, None] = None,
) -> SpacyFastFit:
    if train_dataset is None:
        raise ValueError("train_dataset cannot be None.")
    trainer_args = FastFitTrainerArgs(
        model_name_or_path=model_name_or_path,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        max_text_length=max_text_length,
        dataloader_drop_last=dataloader_drop_last,
        num_repeats=num_repeats,
        optim=optim,
        clf_loss_factor=clf_loss_factor,
        fp16=fp16,
        multi_label=multi_label,
        seed=seed,
    )
    print(trainer_args.train_dataset[:6])
    return SpacyFastFit(
        nlp=nlp,
        trainer_args=trainer_args.model_dump(exclude={"multi_label"}, exclude_none=True),
    )
