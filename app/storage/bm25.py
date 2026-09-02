from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import snowballstemmer

from app.shared.contracts.embedding import SparseVector

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[а-яё]")
_RU_STEMMER = snowballstemmer.stemmer("russian")
_EN_STEMMER = snowballstemmer.stemmer("english")


def _stem(token: str) -> str:
    if _CYRILLIC_RE.search(token):
        return _RU_STEMMER.stemWord(token)
    return _EN_STEMMER.stemWord(token)


_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how in into is it its of on or that
    the this to was were what when where which who will with
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей
    может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз
    тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом
    один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец
    два об другой хоть после над больше тот через эти нас про всего них какая много
    разве эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой
    им более всегда конечно всю между
    """.split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).casefold()
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        tokens.append(_stem(token))
    return tokens


def token_index(token: str, vocab_size: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % (
        vocab_size
    )


@dataclass(frozen=True)
class Bm25Encoder:
    k1: float = 1.2
    b: float = 0.75
    avg_doc_len: int = 256
    vocab_size: int = 1 << 20

    def encode(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        counts: dict[int, int] = {}
        for token in tokens:
            index = token_index(token, self.vocab_size)
            counts[index] = counts.get(index, 0) + 1

        length_norm = self.k1 * (1.0 - self.b + self.b * len(tokens) / self.avg_doc_len)
        indices = sorted(counts)
        values = [counts[i] * (self.k1 + 1.0) / (counts[i] + length_norm) for i in indices]
        return SparseVector(indices=indices, values=values)

    def encode_many(self, texts: list[str]) -> list[SparseVector]:
        return [self.encode(text) for text in texts]
