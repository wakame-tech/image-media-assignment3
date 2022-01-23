import warnings

import nltk

warnings.simplefilter("ignore")
import math
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    return text.lower().split(" ")


syss = [
    "I am a cat",
]
refs = [
    "There is a cat on the mat",
]

syss = list(map(tokenize, syss))
refs = list(map(tokenize, refs))


def n_gram(n: int, words: List[str]) -> List[List[str]]:
    return [words[i : i + n] for i in range(len(words) - n + 1)]


def count(ngram: List[str], words: List[str]) -> int:
    return " ".join(words).count(" ".join(ngram))


def max_ref_count(ngram: List[str], refs: List[List[str]]) -> int:
    return max(count(ngram, ref) for ref in refs)


def count_clip(ngram: List[str], text: List[str], clip: int) -> int:
    return min(clip, count(ngram, text))


def p_n(n: int, cand: List[str], refs: List[List[str]]) -> float:
    match_cnt = 0
    total_cnt = 0
    for ngram in n_gram(n, cand):
        max_ref_count_n = max_ref_count(ngram, refs)
        match_cnt += count_clip(ngram, cand, max_ref_count_n)
    for ngram in n_gram(n, cand):
        total_cnt += count(ngram, cand)
    # print(f"{match_cnt}/{total_cnt}")
    return match_cnt / total_cnt


def my_bleu(N: int, refs: List[List[str]], cand: List[str]) -> float:
    r = max(len(ref) for ref in refs)
    c = len(cand)
    # print(f"c={c}, r={r}")
    bp = math.exp(min(1 - r / c, 0))
    # print(f"bp: {bp}")
    bleu_scores = []
    for n in range(1, N + 1):
        w_n = 1 / n
        pn = p_n(n, cand, refs)
        score = math.exp(w_n * math.log(pn)) if pn > 0 else 0
        bleu_scores.append(score * bp)
    # print(bleu_scores)
    return bleu_scores[-1]


for n in range(1, 4 + 1):
    score = nltk.translate.bleu(refs, syss[0], tuple([1 / n] * n))
    print(f"ref: BLEU-{n}: {score}")
    score = my_bleu(n, refs, syss[0])
    print(f"impl: BLEU-{n}: {score}")
