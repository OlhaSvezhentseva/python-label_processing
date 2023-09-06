#Import Librairies
import re
import jiwer
from nltk import word_tokenize
import argparse
#Import module from this package
from label_postprocessing.vocabulary import extract_vocabulary
from label_postprocessing.utils import load_json, dump_json, read_vocabulary


def get_popular_words(vocabulary: dict, most_frequent: int) -> list[str]:
    """The function extracts the first n-words with the highest occurence from vocabulary."""
    return list(vocabulary.keys())[:most_frequent]


def fix_spelling(labels: list[dict], vocabulary: dict, most_frequent: int, threshold: float) -> None:
    """The function fixes words' spelling in the transcripts if necessary."""
    fixed_labels = []
    popular_words = get_popular_words(vocabulary, most_frequent)
    for label in labels:
        text = label["text"]
        tokens = word_tokenize(text)
        for token in tokens:
            if token.lower() in vocabulary.keys():
                if vocabulary[token.lower()] <= 2:
                    for word in popular_words:
                        cer = jiwer.cer(
                                token.lower(),
                                word
                            )
                        if 0 < cer < threshold:
                            text = re.sub(token, word, text)
        fixed_label = {"ID": label["ID"], "text": text}
        fixed_labels.append(fixed_label)

    dump_json(fixed_labels, "spell_checked_transcripts.json")
    print(f"Saved transcripts in spell_checked_transcripts.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=str)
    parser.add_argument("--freq", type=int)
    parser.add_argument("--dist", type=float)
    parser.add_argument("--voc", nargs='?')
    args = parser.parse_args()
    if not args.voc:
        vocabulary = extract_vocabulary(args.transcripts)
    else:
        vocabulary = read_vocabulary(args.voc)

    labels = load_json(args.transcripts)
    fix_spelling(labels, vocabulary, args.freq, args.dist)

