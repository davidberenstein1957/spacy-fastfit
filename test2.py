from datasets import load_dataset
from fastfit import FastFitTrainer, sample_dataset

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("FastFit/banking_77")
dataset["validation"] = dataset["test"]

# Down sample the train data for 5-shot training
dataset["train"] = sample_dataset(dataset["train"], label_column="label", num_samples_per_label=5)
# del dataset["test"]
del dataset["validation"]
print(dataset)
print(dataset["train"].features)
print(dataset["train"][:5])
trainer = FastFitTrainer(
    model_name_or_path="sentence-transformers/paraphrase-mpnet-base-v2",
    label_column_name="label",
    text_column_name="text",
    num_train_epochs=40,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    max_text_length=128,
    dataloader_drop_last=False,
    num_repeats=4,
    optim="adafactor",
    clf_loss_factor=0.1,
    fp16=True,
    dataset=dataset,
    device="mps",
)

model = trainer.train()
results = trainer.evaluate()

print("Accuracy: {:.1f}".format(results["eval_accuracy"] * 100))
