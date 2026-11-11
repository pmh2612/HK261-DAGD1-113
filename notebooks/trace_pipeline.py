"""Trace one paper and one query through every pipeline stage, for the progress report.

Each stage prints what goes in and what comes out, with real data from the tiny sample
(and SciFact for the evaluation step, since our own queries are not judged yet). With
--out, every stage is also saved as an image (SVG, then PNG via macOS Quick Look).

Run: .venv/bin/python notebooks/trace_pipeline.py [--out ~/Desktop/pipeline-screens]
"""

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from paper_search.config import PROJECT_ROOT, settings
from paper_search.extraction.entities import acronym_aliases, normalize_entity
from paper_search.retrieval import hybrid, rankers

PAPER = "2402.11035"  # Dense Passage Retrieval: Is it Retrieving? (EMNLP 2024 Findings)
QUERY = "retrieval augmented generation for long context question answering"
BGE = "BAAI/bge-base-en-v1.5"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
ALPHA = 0.9  # BT8, tuned on SciFact train
SCIFACT_QID = "94"  # a test claim where BM25 finds the paper and bge-base ranks it 4th

con = Console(record=True, width=112)
OUT: Path | None = None


def step(n: str, title: str, what: str) -> None:
    """Start a stage: a header panel saying what the stage does."""
    con.print(Panel(what, title=f"[bold]BƯỚC {n} · {title}[/]", title_align="left"))


def save(name: str) -> None:
    """Write the stage printed since the last save to OUT/name.png, trimmed."""
    if OUT is None:
        con.export_text()  # clear the record buffer
        return
    svg = OUT / f"{name}.svg"
    con.save_svg(str(svg), title=name)
    size = 2400
    subprocess.run(
        ["qlmanage", "-t", "-s", str(size), "-o", str(OUT), str(svg)], capture_output=True
    )
    # Quick Look centers the drawing in a square: crop the middle back to its aspect ratio.
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text()).groups())
    png = OUT / f"{svg.name}.png"
    crop = [str(round(size * min(h / w, 1))), str(round(size * min(w / h, 1)))]
    subprocess.run(
        ["sips", "-c", *crop, str(png), "--out", str(OUT / f"{name}.png")], capture_output=True
    )
    png.unlink()
    svg.unlink()


def vn(n: int) -> str:
    """12345 -> '12.345' (Vietnamese thousands separator)."""
    return f"{n:,}".replace(",", ".")


def short(text: str, n: int) -> str:
    """Text cut to n characters, with an ellipsis."""
    return text if len(text) <= n else text[: n - 1] + "…"


def jsonl(path: Path) -> list[dict]:
    """Rows of a JSON-lines file."""
    return [json.loads(line) for line in path.open(encoding="utf-8")]


# --- BT1 -------------------------------------------------------------------------------


def frame_stage() -> None:
    """Step 1: listing -> primary-only frame -> shuffled order -> tiny."""
    cfg_months = [f"2024-{m:02d}" for m in range(1, 8)]
    cats = ["cs.IR", "cs.CL", "cs.LG"]
    listed, lists_of = {}, {}
    for c in cats:
        for m in cfg_months:
            for i, primary in json.loads(
                (settings.raw_dir / "arxiv/listing_primary" / f"{c}_{m}.json").read_text()
            ).items():
                listed[i] = primary
                lists_of.setdefault(i, []).append(c)
    frame = sorted(i for i, p in listed.items() if p in cats)
    order = frame.copy()
    random.Random(42).shuffle(order)
    pos = order.index(PAPER)

    step(
        "1",
        "Khung lấy mẫu (BT1)",
        "Đầu vào: trang danh sách theo tháng của arXiv, mỗi bài kèm category chính.\n"
        "Xử lý: gộp 3 category × 7 tháng → chỉ giữ bài có category CHÍNH thuộc 3 nhóm"
        " → sắp xếp → xáo trộn seed 42 → tiny = 200 bài đầu.",
    )
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Bài được liệt kê (có cả cross-list)", vn(len(listed)))
    t.add_row("Khung lấy mẫu (category chính ∈ IR/CL/LG)", vn(len(frame)))
    t.add_row("tiny / small / medium", "200 bài đầu / 2.000 bài đầu / 20.000 bài đầu")
    con.print(t)
    con.print(
        f"\nBài mẫu [bold]{PAPER}[/]: xuất hiện trong danh sách {', '.join(lists_of[PAPER])};"
        f" category chính [bold]{listed[PAPER]}[/] → [green]vào khung[/]."
        f"\nSau khi xáo trộn: vị trí [bold]{pos + 1}[/] / {vn(len(order))}"
        f" → {'[green]thuộc tiny[/]' if pos < 200 else 'không thuộc tiny'}."
    )
    con.print("\n10 vị trí đầu sau khi xáo trộn:", ", ".join(order[:10]))
    save("01_khung_lay_mau")


def sources_stage(paper: dict) -> None:
    """Steps 2-4: raw arXiv XML, raw Semantic Scholar JSON, the merged record."""
    xml = (settings.raw_dir / "arxiv/oai" / f"{PAPER}.xml").read_text()
    body = re.search(r"<arXiv .*?</arXiv>", xml, re.S).group()
    body = re.sub(
        r"(<abstract>)\s*(.*?)\s*(</abstract>)",
        lambda m: m[1] + short(" ".join(m[2].split()), 160) + m[3],
        body,
        flags=re.S,
    )
    body = re.sub(r"\n\s*\n", "\n", body)
    step(
        "2",
        "Lấy dữ liệu từ arXiv (BT1)",
        "Gọi OAI-PMH GetRecord theo arXiv ID (3 giây/request). Phản hồi XML được lưu"
        " cache, chạy lại không gọi mạng.",
    )
    con.print(body, markup=False, highlight=False)
    save("02_arxiv_xml")

    raw = next(
        r
        for r in jsonl(settings.raw_dir / "s2" / "arxiv_tiny_4a8bb592a815.jsonl")
        if (r or {}).get("externalIds", {}).get("ArXiv") == PAPER
    )
    keep = [
        "paperId", "externalIds", "title", "venue", "year", "citationCount",
        "referenceCount", "s2FieldsOfStudy", "authors",
    ]  # fmt: skip
    view = {k: raw.get(k) for k in keep}
    if len(raw["authors"]) > 3:
        view["authors"] = view["authors"][:3] + [f"… {len(raw['authors']) - 3} tác giả nữa"]
    view["abstract"] = short(raw.get("abstract") or "", 70)
    step(
        "3",
        "Lấy dữ liệu từ Semantic Scholar (BT1)",
        "Gọi /paper/batch với 'ARXIV:<id>', 500 bài mỗi request (rút gọn để vừa slide)."
        " Cho những gì arXiv không có: mã tác giả, venue, số trích dẫn, lĩnh vực.",
    )
    con.print_json(json.dumps(view, ensure_ascii=False))
    save("03_semantic_scholar_json")

    rows = [
        ("title", "arXiv", short(paper["title"], 46)),
        ("abstract", "arXiv", short(paper["abstract"], 46)),
        ("year", "S2 (năm xuất bản)", str(paper["year"])),
        ("venue", "S2 (bỏ 'arXiv.org')", short(paper["venue"] or "—", 46)),
        ("authors", "S2", f"{len(paper['authors'])} tác giả, kèm mã S2"),
        ("doi", "arXiv, thiếu thì S2", paper["doi"] or "—"),
        ("citation_count", "S2", str(paper["citation_count"])),
        ("fields_of_study", "S2", ", ".join(paper["fields_of_study"] or [])),
        ("categories", "arXiv", " ".join(paper["categories"])),
    ]
    step(
        "4",
        "Ghép hai nguồn → 1 bản ghi (BT1)",
        "Ghép theo arXiv ID. arXiv là bản gốc cho nội dung; Semantic Scholar bổ sung metadata."
        " Kết quả là 1 dòng trong papers_tiny.jsonl.",
    )
    t = Table("Trường", "Lấy từ", "Giá trị trong bản ghi cuối")
    for r in rows:
        t.add_row(*r)
    con.print(t)
    save("04_ghep_ban_ghi")


# --- BT6 / BT7 / BT8 -------------------------------------------------------------------


def search_stage(docs: list[dict], j: int) -> None:
    """Steps 5-8: BM25 analysis and scores, bge-base vectors, linear fusion."""
    analyze = rankers.analyzer("stop_stem")
    text = rankers.FIELDS["title2_abstract"](docs[j])
    lower = re.findall(r"\w+", text.lower())
    stop = rankers.analyzer("stop")(text)
    stem = analyze(text)
    step(
        "5",
        "Index BM25: tiền xử lý văn bản (BT6)",
        "Văn bản index = title ×2 + abstract. Cùng một cách xử lý cho bài báo và cho query:"
        " lowercase → bỏ stopword → stemming.",
    )
    t = Table("Giai đoạn", "Số token", "20 token đầu")
    t.add_row("Tách từ + lowercase", str(len(lower)), " ".join(lower[:20]))
    t.add_row("Bỏ stopword", str(len(stop)), " ".join(stop[:20]))
    t.add_row("Stemming (Snowball)", str(len(stem)), " ".join(stem[:20]))
    con.print(t)
    con.print(f"\nQuery: [italic]{QUERY}[/]\n→ token: [bold]{' '.join(analyze(QUERY))}[/]")
    save("05_bm25_tien_xu_ly")

    corpus = [analyze(rankers.FIELDS["title2_abstract"](d)) for d in docs]
    index = rankers.BM25Okapi(corpus, k1=1.5, b=0.75)
    bm = index.get_scores(analyze(QUERY))
    chatqa = max(range(len(docs)), key=lambda i: bm[i])
    step(
        "6",
        "Chấm điểm BM25 cho query (BT6)",
        "Điểm = tổng theo từng từ của query: idf(từ) × tf·(k1+1) / (tf + k1·(1−b+b·dl/avgdl)),"
        " k1 = 1,5, b = 0,75.\nTừ hiếm (idf cao) và xuất hiện nhiều trong bài (tf cao) thì"
        " đóng góp nhiều.",
    )
    t = Table("Từ trong query", "idf", "tf bài mẫu", "điểm bài mẫu", "tf ChatQA 2", "điểm ChatQA 2")

    def part(term: str, i: int) -> tuple[int, float]:
        tf = index.doc_freqs[i].get(term, 0)
        dl = index.doc_len[i]
        idf = index.idf.get(term, 0)
        return tf, idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / index.avgdl))

    for term in analyze(QUERY):
        tf_a, s_a = part(term, j)
        tf_b, s_b = part(term, chatqa)
        t.add_row(
            term, f"{index.idf.get(term, 0):.2f}", str(tf_a), f"{s_a:.2f}", str(tf_b), f"{s_b:.2f}"
        )
    t.add_row("[bold]Tổng[/]", "", "", f"[bold]{bm[j]:.2f}[/]", "", f"[bold]{bm[chatqa]:.2f}[/]")
    con.print(t)
    con.print(
        f"Bài mẫu: [bold]{short(docs[j]['title'], 50)}[/]   ChatQA 2: [bold]"
        f"{short(docs[chatqa]['title'], 50)}[/] (BM25 hạng 1 cho query này)"
    )
    save("06_bm25_cham_diem")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BGE)
    doc_vecs = model.encode(
        [f"{d['title']}. {d['abstract']}" for d in docs], normalize_embeddings=True
    )
    q_vec = model.encode([BGE_PREFIX + QUERY], normalize_embeddings=True)[0]
    cos = doc_vecs @ q_vec
    step(
        "7",
        "Index embedding bge-base (BT7)",
        "Mỗi bài: 'title. abstract' → vector 768 chiều, chuẩn hóa độ dài = 1, lưu trong FAISS."
        "\nQuery thêm câu lệnh mà bge yêu cầu ở đầu, rồi cũng thành vector 768 chiều."
        " Điểm = cosine giữa hai vector.",
    )
    fmt = lambda v: "[" + ", ".join(f"{x:+.3f}" for x in v[:5]) + ", …]"  # noqa: E731
    t = Table("", "Văn bản đưa vào model", "Vector (5/768 chiều đầu)")
    t.add_row("Bài mẫu", short(f"{docs[j]['title']}. {docs[j]['abstract']}", 52), fmt(doc_vecs[j]))
    t.add_row("Query", short(BGE_PREFIX + QUERY, 52), fmt(q_vec))
    con.print(t)
    con.print(
        f"‖vector‖ = {np.linalg.norm(doc_vecs[j]):.3f}   cosine(query, bài mẫu) ="
        f" [bold]{cos[j]:.3f}[/]   cosine(query, ChatQA 2) = [bold]{cos[chatqa]:.3f}[/]"
    )
    save("07_embedding")

    bm_n, cos_n = hybrid.minmax(bm[None])[0], hybrid.minmax(cos[None])[0]
    fused = hybrid.linear(bm[None], cos[None], ALPHA)[0]

    def rank(s: np.ndarray) -> dict[int, int]:
        return {int(i): r + 1 for r, i in enumerate(np.argsort(-s, kind="stable"))}

    r_bm, r_cos, r_fu = rank(bm), rank(cos), rank(fused)
    step(
        "8",
        "Tìm kiếm lai: kết hợp tuyến tính (BT8)",
        f"Mỗi loại điểm đưa về [0, 1] theo query (min-max), rồi: {ALPHA} × embedding +"
        f" {1 - ALPHA:.1f} × BM25. α = {ALPHA} chọn trên tập train SciFact.\n"
        f"Query: [italic]{QUERY}[/]  ·  corpus: 200 bài tiny",
    )
    t = Table(
        "Hạng", "Bài báo", "BM25", "→[0,1]", "Hạng BM25", "cosine", "→[0,1]", "Hạng bge", "Điểm lai"
    )
    for i in np.argsort(-fused, kind="stable")[:8]:
        mark = "[bold yellow]" if i in (j, chatqa) else ""
        t.add_row(
            str(r_fu[i]),
            f"{mark}{short(docs[i]['title'], 30)}",
            f"{bm[i]:.2f}",
            f"{bm_n[i]:.2f}",
            str(r_bm[i]),
            f"{cos[i]:.3f}",
            f"{cos_n[i]:.2f}",
            str(r_cos[i]),
            f"[bold]{fused[i]:.3f}[/]",
        )
    con.print(t)
    con.print(
        "Hai hệ không đồng ý: BM25 đưa ChatQA 2 (khớp đúng 'long context', 'RAG') lên đầu,"
        " bge-base đưa bài mẫu lên đầu.\nĐúng hay sai cần nhãn liên quan 0/1/2 của người"
        " → lý do phải có bộ 50 query tự gán nhãn."
    )
    save("08_tim_kiem_lai")


# --- BT2 / BT4 / BT5 -------------------------------------------------------------------


def extraction_stage(paper: dict) -> list[tuple[str, str]]:
    """Step 9: SciBERT tags every word of the abstract; returns (name, type)."""
    import spacy
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from paper_search.extraction.ner import decode_bio, device

    nlp = spacy.load("en_core_web_sm")
    sents = [[t.text for t in s if not t.is_space] for s in nlp(paper["abstract"]).sents]
    model_dir = settings.processed_dir / "models/scibert_scierc"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device()).eval()

    def tag(words: list[str]) -> list[str]:
        enc = tok(words, is_split_into_words=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            ids = model(**enc.to(model.device)).logits[0].argmax(-1).tolist()
        tags, seen = ["O"] * len(words), set()
        for pos, w in enumerate(enc.word_ids()):
            if w is not None and w not in seen:
                seen.add(w)
                tags[w] = model.config.id2label[ids[pos]]
        return tags

    entities, shown = [], None
    for words in sents:
        tags = tag(words)
        found = [(" ".join(words[a : b + 1]), t) for a, b, t in decode_bio(tags)]
        entities += found
        if shown is None and len(found) >= 3:
            shown = (words, tags)

    step(
        "9",
        "Trích xuất thực thể bằng SciBERT (BT2)",
        "Abstract → tách câu, tách từ → SciBERT (đã fine-tune trên SciERC) gắn nhãn cho"
        " từng từ: B- = chữ đầu thực thể, I- = chữ tiếp theo, O = không phải thực thể.",
    )
    words, tags = shown
    t = Table(show_header=False, box=None, padding=(0, 1))
    line_w, line_t = [], []
    for w, g in zip(words, tags, strict=True):
        style = "bold green" if g != "O" else "dim"
        line_w.append(f"[{style}]{w}[/]")
        line_t.append(f"[{style}]{g}[/]")
    for k in range(0, len(words), 9):
        t.add_row(*line_w[k : k + 9])
        t.add_row(*line_t[k : k + 9])
        t.add_row("")
    con.print("Một câu trong abstract, từng từ và nhãn:")
    con.print(t)
    by_type: dict[str, list[str]] = {}
    for name, ty in entities:
        by_type.setdefault(ty, []).append(name)
    t = Table("Loại", f"Thực thể tìm được trong cả abstract ({len(entities)})")
    for ty in ("Task", "Method", "Metric", "Material"):
        t.add_row(ty, "; ".join(dict.fromkeys(by_type.get(ty, []))) or "—")
    con.print(t)
    save("09_trich_xuat_thuc_the")
    return entities


def normalize_stage(paper: dict, entities: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Step 10: entity names to node keys; returns (name, type, key)."""
    aliases = acronym_aliases([paper["abstract"]])
    step(
        "10",
        "Chuẩn hóa tên thực thể (BT4, phương pháp luật)",
        "Tên → khóa chuẩn: bỏ phần trong ngoặc, lowercase, bỏ hậu tố như '-base',"
        " 'method'; cụm quá chung chung ('our model', 'data') bị loại.\nViết tắt được gộp với"
        " tên đầy đủ nếu abstract có định nghĩa dạng 'Dense Passage Retrieval (DPR)'.",
    )
    t = Table("Tên trong abstract", "Loại", "Khóa sau chuẩn hóa", "Kết quả")
    rows, seen = [], set()
    for name, ty in entities:
        key = normalize_entity(name)
        key = aliases.get(key, key)
        if (name, ty) in seen:
            continue
        seen.add((name, ty))
        verdict = "[red]loại (chung chung)[/]" if not key else "giữ"
        if key and key != name.lower():
            verdict = "giữ, đã đổi tên"
        t.add_row(name, ty, key or "—", verdict)
        if key:
            rows.append((name, ty, key))
    con.print(t)
    con.print(f"Viết tắt tìm được trong abstract: {aliases or 'không có'}")
    save("10_chuan_hoa")
    return rows


def graph_stage(paper: dict, rows: list[tuple[str, str, str]]) -> None:
    """Step 11: the Cypher this paper would become (BT3 and BT5 not built yet)."""
    label = {"Task": "Task", "Method": "Method", "Metric": "Metric", "Material": "Dataset"}
    step(
        "11",
        "Nạp vào Knowledge Graph (BT5, minh họa, CHƯA nạp)",
        "Mọi lệnh là MERGE theo khóa duy nhất → chạy lại không tạo bản sao.\n"
        "Metadata (tác giả, venue, lĩnh vực) đi thẳng từ BT1. Quan hệ bài báo → thực thể"
        " cần BT3 để biết là PROPOSES, USES, EVALUATES_ON… nên tạm ghi là MENTIONS.",
    )
    lines = [
        f"MERGE (p:Paper {{paper_id: '{PAPER}'}}) SET p.title = '{short(paper['title'], 40)}',"
        f" p.year = {paper['year']}",
        f"MERGE (v:Venue {{name: '{short(paper['venue'] or '', 50)}'}})",
        "MERGE (p)-[:PUBLISHED_IN]->(v)",
    ]
    for name, aid in list(zip(paper["authors"], paper["author_ids"], strict=True))[:2]:
        lines.append(f"MERGE (:Author {{author_id: '{aid}', name: '{name}'}})-[:AUTHORED]->(p)")
    if len(paper["authors"]) > 2:
        lines.append(f"// … {len(paper['authors']) - 2} tác giả nữa")
    for f in paper["fields_of_study"] or []:
        lines.append(f"MERGE (p)-[:HAS_FIELD]->(:FieldOfStudy {{name: '{f}'}})")
    for ty, key in list(dict.fromkeys((ty, key) for _, ty, key in rows))[:6]:
        lines.append(
            f"MERGE (p)-[:MENTIONS]->(:{label[ty]} {{key: '{key}'}})   // BT3 sẽ đặt tên quan hệ"
        )
    con.print("\n".join(lines), markup=False, highlight=False)
    save("11_knowledge_graph")


# --- evaluation ------------------------------------------------------------------------


def eval_stage() -> None:
    """Step 12: nDCG@10 of one judged SciFact query, for each system."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_search import load

    docs, queries, qrels, _ = load("scifact")
    q = queries[SCIFACT_QID]
    rel = qrels[SCIFACT_QID]
    bm = rankers.bm25_scores(docs, "title2_abstract", "stop_stem")([q])
    cos = rankers.dense_scores(docs, BGE, BGE_PREFIX)([q])
    fused = hybrid.linear(bm, cos, ALPHA)
    step(
        "12",
        "Đánh giá một query (BT6–BT8, SciFact)",
        "Bộ query của nhóm chưa có nhãn, nên minh họa bằng một query SciFact có nhãn sẵn.\n"
        "nDCG@10 = DCG / DCG lý tưởng, với DCG = Σ độ liên quan / log2(hạng + 1) trên"
        " 10 kết quả đầu. Điểm cuối = trung bình trên 300 query test.",
    )
    con.print(f"Query {SCIFACT_QID}: [italic]{q}[/]   ·  bài liên quan: {list(rel)}")
    t = Table("Hệ", "Hạng của bài liên quan", "DCG", "nDCG@10")
    for name, s in (("BM25", bm[0]), ("bge-base", cos[0]), (f"Lai α = {ALPHA}", fused[0])):
        order = [docs[i]["paper_id"] for i in np.argsort(-s, kind="stable")]
        ranks = [order.index(p) + 1 for p in rel]
        dcg = sum(rel[p] / math.log2(r + 1) for p, r in zip(rel, ranks, strict=True) if r <= 10)
        ideal = sum(g / math.log2(k + 2) for k, g in enumerate(sorted(rel.values(), reverse=True)))
        t.add_row(name, ", ".join(map(str, ranks)), f"{dcg:.3f}", f"[bold]{dcg / ideal:.3f}[/]")
    con.print(t)
    con.print(
        "Tên thuốc hiếm 'Albendazole' là từ khóa BM25 bắt được ngay; embedding xếp bài"
        " đúng thấp hơn.\nKết hợp giữ được cả hai → trên 300 query: 0,763 so với bge-base 0,742"
        " (p = 0,0015)."
    )
    save("12_danh_gia")


def main() -> None:
    """Run every stage in pipeline order."""
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    OUT = parser.parse_args().out
    if OUT:
        OUT = OUT.expanduser()
        OUT.mkdir(parents=True, exist_ok=True)

    papers = jsonl(settings.processed_dir / "papers_tiny.jsonl")
    j = next(k for k, p in enumerate(papers) if p["paper_id"] == PAPER)
    frame_stage()
    sources_stage(papers[j])
    search_stage(papers, j)
    entities = extraction_stage(papers[j])
    rows = normalize_stage(papers[j], entities)
    graph_stage(papers[j], rows)
    eval_stage()


if __name__ == "__main__":
    main()
