"""BM25 lexical vectors for the sparse retrieval branch (Qdrant's built-in BM25).

Qdrant owns the IDF half of BM25: a sparse vector index declared with
``modifier: idf`` multiplies each term by the inverse document frequency it observes
across the collection at query time. This module produces the other half — the term
frequency component with length normalisation — so the lexical branch needs no model
server at all:

    value(t, D) = tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))

Tokens are mapped to sparse indices by a stable hash, exactly as Qdrant's own BM25
encoder does; the space is large enough (2^20 by default) that collisions are noise.

Queries and documents run through the same encoder. A query term normally occurs once,
so its weights differ from 1.0 only by a factor that is constant for the whole query and
therefore cannot change the ranking between documents.

Snowball stemming (RU for Cyrillic tokens, EN otherwise) is applied identically on both
sides so inflected query forms match document forms — «стоимости» ↔ «стоимость»,
«процедурах» ↔ «процедур» — which raw hashing would miss (search-improvement S1, Tier 2.1).
Numbers and identifiers carry no suffixes, so the stemmer passes them through unchanged and
the branch still catches exact terminology a dense model blurs.
"""

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
    """Stem a casefolded token: Russian stemmer for Cyrillic, English otherwise."""
    if _CYRILLIC_RE.search(token):
        return _RU_STEMMER.stemWord(token)
    return _EN_STEMMER.stemWord(token)


# Only the highest-frequency function words: IDF already discounts common terms, so this
# list exists to keep vectors small rather than to do linguistics.
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
    """Stemmed word tokens, minus single characters and the stopword list.

    Stopwords are removed on their full surface form (the list is full forms) before
    stemming; numbers pass through the stemmer unchanged.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).casefold()
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        tokens.append(_stem(token))
    return tokens


def token_index(token: str, vocab_size: int) -> int:
    """Stable 32-bit hash of a token, folded into the configured sparse dimension."""
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
