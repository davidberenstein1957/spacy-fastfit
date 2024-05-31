import spacy
import spacy_fastfit  # noqa

nlp = spacy.blank("en")

train_dataset = {
    "inlier": ["This text is about chairs.", "Couches, benches and televisions.", "I really need to get a new sofa."],
    "outlier": ["Text about kitchen equipment", "This text is about politics", "Comments about AI and stuff."],
}


nlp.add_pipe(
    "spacy_fastfit",
    config={"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2", "train_dataset": train_dataset},
)
