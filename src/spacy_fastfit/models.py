import logging
import types
from typing import Union

from datasets import DatasetDict
from fastfit import FastFit, FastFitTrainer
from spacy import util
from spacy.language import Language
from spacy.tokens import Doc
from transformers import pipeline

__LOGGER__ = logging.getLogger(__name__)


class SpacyFastFit:
    def __init__(self, nlp: Language, trainer_args: dict):
        print(trainer_args)
        dataset_dict = {}
        if "train_dataset" in trainer_args:
            dataset_dict["train"] = trainer_args["train_dataset"]
            del trainer_args["train_dataset"]
        if "eval_dataset" in trainer_args:
            dataset_dict["validation"] = trainer_args["eval_dataset"]
            del trainer_args["eval_dataset"]
        trainer_args["dataset"] = DatasetDict(dataset_dict)
        self.train(trainer_args)

    def train(cls, trainer_args: dict):
        trainer = FastFitTrainer(**trainer_args)
        model = trainer.train()
        if "eval_dataset" in trainer_args:
            results = trainer.evaluate()
            print("Accuracy: {:.1f}".format(results["eval_accuracy"] * 100))
        cls.from_pretrained(model, tokenizer=trainer.tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        nlp: Language,
        model_name_or_path: Union[str, FastFit],
    ):
        cls.nlp = nlp
        if isinstance(model_name_or_path, str):
            model = FastFit.from_pretrained(model_name_or_path)
        tokenizer = model.tokenizer
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        cls.classifier = classifier
        return cls

    def __call__(self, doc: Doc):
        """
        It takes a doc, gets the embeddings from the doc, reshapes the embeddings, gets the prediction from the embeddings,
        and then sets the prediction results for the doc

        :param doc: Doc
        :type doc: Doc
        :return: The doc object with the predicted categories and the predicted categories for each sentence.
        """
        if isinstance(doc, str):
            doc = self.nlp(doc)
        prediction = self.model.predict_proba([doc.text])
        doc = self._assign_labels(doc, prediction[0])

        return doc

    def pipe(self, stream, batch_size=128, include_sent=None):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        if isinstance(stream, str):
            stream = [stream]

        if not isinstance(stream, types.GeneratorType):
            stream = self.nlp.pipe(stream, batch_size=batch_size)

        for docs in util.minibatch(stream, size=batch_size):
            pred_results = self.model.predict_proba([doc.text for doc in docs])

            for doc, prediction in zip(docs, pred_results):
                yield self._assign_labels(doc, prediction)
