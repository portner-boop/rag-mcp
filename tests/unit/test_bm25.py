from __future__ import annotations

from app.storage.bm25 import Bm25Encoder, tokenize


def test_tokenizer_keeps_words_and_numbers_and_drops_stopwords() -> None:
    tokens = tokenize("Сколько дней отпуска в 2026 году?")
    assert "отпуск" in tokens and "2026" in tokens
    assert "в" not in tokens


def test_stemming_collapses_russian_inflections() -> None:
    assert tokenize("стоимости") == tokenize("стоимость")
    assert tokenize("процедурах") == tokenize("процедур")
    assert tokenize("эксплуатации") == tokenize("эксплуатация")


def test_encoding_is_deterministic() -> None:
    encoder = Bm25Encoder()
    first = encoder.encode("командировочные расходы и суточные")
    second = encoder.encode("командировочные расходы и суточные")
    assert (first.indices, first.values) == (second.indices, second.values)


def test_term_frequency_saturates() -> None:
    encoder = Bm25Encoder()
    once = encoder.encode("отпуск " + "filler " * 20)
    twice = encoder.encode("отпуск отпуск " + "filler " * 20)
    term = encoder.encode("отпуск").indices[0]
    single = dict(zip(once.indices, once.values, strict=True))[term]
    double = dict(zip(twice.indices, twice.values, strict=True))[term]
    assert single < double < 2 * single


def test_longer_documents_are_penalised_for_the_same_term() -> None:
    encoder = Bm25Encoder()
    term = encoder.encode("политика").indices[0]
    short = _weights(encoder.encode("политика отпусков"))[term]
    long = _weights(encoder.encode("политика отпусков " + "прочее " * 300))[term]
    assert long < short


def test_query_terms_share_one_weight_so_ranking_is_unaffected() -> None:
    vector = Bm25Encoder().encode("суточные за рубежом")
    assert len(set(round(v, 12) for v in vector.values)) == 1


def test_text_without_terms_yields_an_empty_vector() -> None:
    vector = Bm25Encoder().encode("и в на о")
    assert vector.indices == [] and vector.values == []


def test_indices_stay_inside_the_configured_vocabulary() -> None:
    encoder = Bm25Encoder(vocab_size=4096)
    vector = encoder.encode(" ".join(f"term{i}" for i in range(500)))
    assert all(0 <= i < 4096 for i in vector.indices)
    assert vector.indices == sorted(vector.indices)


def _weights(vector) -> dict[int, float]:
    return dict(zip(vector.indices, vector.values, strict=True))
