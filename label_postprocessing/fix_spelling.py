
import re
import jiwer
from nltk import word_tokenize
from utils import load_json, dump_json, read_vocabulary

def get_popular_words(vocabulary, most_frequent):
    return list(vocabulary.keys())[:most_frequent]


def fix_spelling(labels, vocabulary, most_frequent, threshold):
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



labels = load_json("corrected_transcripts.json")
vocabulary = read_vocabulary('vocabulary.csv')
print(fix_spelling(labels, vocabulary, 20, 0.34))
