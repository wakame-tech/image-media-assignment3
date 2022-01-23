import nltk

nltk.download("wordnet")
from nltk.translate.meteor_score import meteor_score

syss = "I am a cat"
refs = [
    "There is a cat on the mat",
]
scorer = meteor_score(refs, syss)
print(scorer)
