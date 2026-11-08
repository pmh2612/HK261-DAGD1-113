"""Hand-computed cases for the BT2 metrics and the SciERC loader."""

import json

from paper_search.eval.ner import load_scierc, score
from paper_search.extraction.ner import decode_bio, locate


def test_exact_and_partial_scores():
    gold = {"d": [(0, 10, "Task"), (20, 25, "Method"), (30, 35, "Metric")]}
    pred = {"d": [(0, 10, "Task"), (20, 23, "Method"), (30, 35, "Method"), (40, 45, "Task")]}
    s = score(gold, pred)
    # exact: only the Task span matches -> P 1/4, R 1/3
    assert s["all"]["exact"]["p"] == 0.25 and s["all"]["exact"]["r"] == round(1 / 3, 4)
    # partial: Task + the truncated Method count; wrong-type Metric and spurious Task do not
    assert s["all"]["partial"]["p"] == 0.5 and s["all"]["partial"]["r"] == round(2 / 3, 4)
    assert s["Metric"]["exact"]["r"] == 0.0 and s["Task"]["exact"]["p"] == 0.5


def test_types_outside_the_four_are_ignored():
    s = score({"d": [(0, 3, "Generic")]}, {"d": [(0, 3, "Generic")]})
    assert s["all"]["gold"] == 0 and s["all"]["pred"] == 0


def test_load_scierc_maps_word_spans_to_characters(tmp_path):
    row = {
        "doc_key": "X1",
        "sentences": [["We", "use", "CRFs", "-LRB-", "1", "-RRB-", "."], ["BLEU", "rises"]],
        "ner": [[[2, 2, "Method"]], [[7, 7, "Metric"]]],
        "relations": [[], []],
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(row) + "\n")
    (doc,) = load_scierc(path)
    assert doc["text"] == "We use CRFs ( 1 ) . BLEU rises"
    assert [doc["text"][s:e] for s, e, _ in doc["entities"]] == ["CRFs", "BLEU"]
    assert doc["sentences"] == [(0, 7), (7, 9)]


def test_decode_bio_and_locate():
    assert decode_bio(["B-Task", "I-Task", "O", "I-Method", "B-Metric"]) == [
        (0, 1, "Task"),
        (3, 3, "Method"),  # a stray I- starts a new entity rather than being dropped
        (4, 4, "Metric"),
    ]
    text = "parsing with CRFs ; crfs beat parsing ."
    assert locate(text, "CRFs", "Method") == [(13, 17, "Method")]
    assert locate(text, "crf", "Method") == []  # whole words only
    assert locate(text, "PARSING", "Task") == [(0, 7, "Task"), (30, 37, "Task")]


def test_errors_are_classified():
    from paper_search.eval.ner import errors

    gold = {"d": [(0, 10, "Task"), (20, 25, "Method"), (50, 55, "Metric")]}
    pred = {"d": [(0, 10, "Task"), (20, 23, "Method"), (50, 55, "Method"), (70, 75, "Task")]}
    e = errors(gold, pred)
    assert [s for _, s in e["boundary"]] == [(20, 23, "Method")]
    assert [s for _, s in e["type"]] == [(50, 55, "Method")]
    assert [s for _, s in e["spurious"]] == [(70, 75, "Task")]
    assert e["missed"] == []  # the Metric is touched, by a wrong-type prediction


def test_bootstrap_p_separates_clear_and_null_differences():
    from paper_search.eval.ner import bootstrap_p

    gold = {f"d{i}": [(0, 5, "Task"), (10, 15, "Method")] for i in range(30)}
    perfect = {k: list(v) for k, v in gold.items()}
    half = {k: v[:1] for k, v in gold.items()}
    diff, p = bootstrap_p(gold, perfect, half, rounds=2000)
    assert diff == round(1 - 2 / 3, 4) and p == 0.0
    diff, p = bootstrap_p(gold, perfect, perfect, rounds=200)
    assert diff == 0.0 and p == 1.0


def test_parse_answer_keeps_bare_strings_whole():
    from paper_search.extraction.ner import parse_answer

    parsed, not_a_list = parse_answer(
        'Sure: {"Task": "3-D stereo reconstruction", "Method": ["CRF", "CRF", " "], '
        '"Metric": "", "Material": []}'
    )
    assert parsed["Task"] == ["3-D stereo reconstruction"]  # not 26 characters
    # "" means nothing found, so only the Task string counts as a format slip
    assert parsed["Method"] == ["CRF"] and parsed["Metric"] == [] and not_a_list == 1
    assert parse_answer("no json here") == (None, 0)


def test_locate_ignores_spacing_around_punctuation():
    text = "we train Convolutional Neural Networks ( CNN ) and CNNs ."
    assert locate(text, "Convolutional Neural Networks (CNN)", "Method") == [(9, 46, "Method")]
    assert locate(text, "CNN", "Method") == [(41, 44, "Method")]  # not inside "CNNs"
    assert locate(text, "Neural Networks", "Method") == [(23, 38, "Method")]
    assert locate(text, "NeuralNetworks", "Method") == []  # words still need a space
