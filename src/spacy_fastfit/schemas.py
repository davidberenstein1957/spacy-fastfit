import logging
from typing import Union

import pandas as pd
from datasets import Dataset, Value
from pydantic import BaseModel, model_validator

__LOGGER__ = logging.getLogger(__name__)


class FastFitTrainerArgs(BaseModel):
    """
    SetFitTrainerArgs is a Pydantic model that defines the arguments for the SetFitTrainer.
    __NOTE__: it is a simplified version of the official args from the SetFit library.
    model_name_or_path: str
    train_dataset: Union[dict, Dataset]
    test_dataset: Union[dict, Dataset]
    validation_dataset: Union[dict, Dataset] = None
    text_column_name="text"
    label_column_name="label"
    num_train_epochs=40
    per_device_train_batch_size=2
    per_device_eval_batch_size=2
    max_text_length=512
    dataloader_drop_last=False
    num_repeats=4
    optim="adafactor"
    clf_loss_factor=0.1
    fp16=True
    multi_label=False
    seed=None
    """

    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    train_dataset: Union[dict, Dataset, None]
    test_dataset: Union[dict, Dataset, None]
    validation_dataset: Union[dict, Dataset, None] = None
    label_column_name: str = "label"
    text_column_name: str = "text"
    num_train_epochs: int = 40
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    max_text_length: int = 512
    dataloader_drop_last: bool = False
    num_repeats: int = 4
    optim: str = "adafactor"
    clf_loss_factor: float = 0.1
    fp16: bool = True
    multi_label: bool = False
    seed: Union[int, None] = None
    do_predict: bool = False
    task_name: bool = None

    class Config:
        arbitrary_types_allowed = True
        fields = {"multi_label": {"exclude": True}, "label": {"exclude": True}}

    @model_validator(mode="before")
    def convert_dict_to_dataset(cls, values):
        def _convert_dict_to_dataset(ds_dict):
            df = pd.DataFrame(datasets_dict)
            labels = df.label.unique().tolist()
            df_duplicate_in_group = df.drop_duplicates(subset=["text", "label"])
            df_duplicates_across_groups = df.drop_duplicates(subset=["text", "label", "split"])
            if (len(df_duplicate_in_group) != len(df)) and (
                len(df_duplicate_in_group) != len(df_duplicates_across_groups)
            ):
                __LOGGER__.warning("There are duplicate texts acrooss the train and eval data.")
            elif len(df_duplicate_in_group) != len(df):
                __LOGGER__.warning("There are duplicate texts in the dataset.")
            df = df.drop_duplicates(subset=["text", "label", "split"])
            df_group = df.groupby("text").agg(list).reset_index()
            if len(df_group) != len(df):
                values["multi_label"] = True
                df = df_group
            return df, labels

        def _create_datasets(df: pd.DataFrame, labels):
            features = {
                "text": Value(dtype="string", id=None),
                "label": Value(dtype="string", id=None),
            }
            ds = Dataset.from_dict(df.to_dict(orient="list"))
            ds = ds.shuffle(seed=values["seed"])
            return ds

        def _add_data_to_dict(datasets_dict, data, train_or_test):
            for label, texts in data.items():
                for text in texts:
                    datasets_dict["text"].append(text)
                    datasets_dict["label"].append(label)
                    datasets_dict["split"].append(train_or_test)
            return datasets_dict

        if isinstance(values["train_dataset"], dict):
            options = ["train_dataset"]
            datasets_dict = {"text": [], "label": [], "split": []}
            datasets_dict = _add_data_to_dict(datasets_dict, values["train_dataset"], "train_dataset")
            if isinstance(values["validation_dataset"], dict):
                options.append("validation_dataset")
                datasets_dict = _add_data_to_dict(datasets_dict, values["validation_dataset"], "validation_dataset")
            elif isinstance(values["validation_dataset"], Dataset):
                raise ValueError("train_dataset and validation_dataset must be of the same type")

            df, labels = _convert_dict_to_dataset(datasets_dict)
            values["label"] = labels
            text = ""
            for train_or_test in options:
                if values["multi_label"]:
                    df_filtered = df.copy(deep=True)
                    df_filtered["split"] = df_filtered["split"].apply(lambda x: True if train_or_test in x else False)
                    df_filtered = df_filtered[df_filtered["split"] == True]  # noqa
                else:
                    df_filtered = df[df["split"] == train_or_test]

                if not df_filtered.empty:
                    df_filtered = df_filtered.drop(columns=["split"])
                    values[train_or_test] = _create_datasets(df_filtered, labels)
                    text += f"\n\t{train_or_test}: {len(values[train_or_test])}"
            __LOGGER__.info(
                f"The datasets have been created: \n\tlabels: {values['label']}\n\tmulti_label: {values['multi_label']}{text}"
            )

            return values
        else:
            return values
