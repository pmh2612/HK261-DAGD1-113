# Kế hoạch thực nghiệm — Hệ thống tìm kiếm bài báo khoa học (KG + Hybrid Search)

> Tài liệu này dành cho Claude Code. Đọc toàn bộ trước khi viết code. Mọi quyết định kỹ thuật phải bám theo các nguyên tắc ở mục 1.

---

## 0. Bối cảnh

- Đồ án 2 người (Hiếu, Hà My), Giai đoạn 1: 09–12/2026. Giai đoạn 2 (RAG, chatbot, UI) làm ở học kỳ sau — **ngoài phạm vi tài liệu này**.
- Mục tiêu Giai đoạn 1: chia hệ thống thành các bài toán nhỏ, với mỗi bài toán **thử nhiều phương pháp/mô hình, đo bằng thực nghiệm, chọn phương pháp tốt nhất** kèm số liệu chứng minh.
- Stack: Python 3.11+, Semantic Scholar API, arXiv API, Neo4j (Cypher), sentence-transformers, FAISS, rank_bm25.

## 1. Nguyên tắc bắt buộc (yêu cầu của giảng viên hướng dẫn)

1. **Chỉ dùng abstract** (+ metadata: title, authors, year, venue, citations, fields of study). Không parse full-text PDF.
2. **Không xử lý hình ảnh.** Nếu một bài thiếu abstract trong API, chỉ dùng cách đơn giản nhất: tải PDF, lấy text trang 1 bằng PyMuPDF (`page.get_text()`), cắt đoạn giữa "Abstract" và "Introduction". Không OCR ảnh, không layout model. Nếu vẫn thất bại → bỏ bài đó và ghi log.
3. **Nhỏ trước, rộng sau.** Mọi thí nghiệm chạy trên sample nhỏ trước. Chỉ mở rộng khi kết quả sample đã được người dùng xem và đồng ý.
4. **Mọi kết luận phải có số liệu.** Không chọn phương pháp theo cảm tính. Mỗi bài toán con có metric, tập đánh giá cố định và bảng so sánh.
5. **Tái lập được.** Cố định seed, version dữ liệu, version model; cấu hình bằng file YAML; kết quả ghi ra file.

## 2. Các mức dữ liệu (scale-up)

| Mức | Số bài | Mục đích |
|---|---|---|
| `tiny` | 200 | Debug pipeline, gán nhãn thủ công, so sánh nhanh nhiều phương pháp |
| `small` | 2.000 | So sánh nghiêm túc, chọn phương pháp tốt nhất |
| `medium` | 20.000 | Kiểm tra phương pháp được chọn có giữ chất lượng và tốc độ khi mở rộng |

- Domain cố định để dễ gán nhãn: arXiv `cs.IR`, `cs.CL`, `cs.LG` (có thể điều chỉnh).
- `tiny ⊂ small ⊂ medium` (lấy mẫu lồng nhau, cùng seed) để kết quả so sánh được giữa các mức.
- **Dừng lại hỏi người dùng** trước khi chuyển từ mức này sang mức tiếp theo.

## 3. Cấu trúc repo đề xuất

```
.
├── configs/                # YAML cho từng thí nghiệm
├── data/
│   ├── raw/                # JSON gốc từ API (không sửa)
│   ├── processed/          # papers_{tiny,small,medium}.jsonl
│   └── annotations/        # nhãn thủ công (gold) cho đánh giá
├── src/
│   ├── collect/            # gọi API, phân trang, rate limit, cache
│   ├── extract/            # trích xuất thực thể/quan hệ
│   ├── normalize/          # chuẩn hóa thực thể
│   ├── kg/                 # schema + nạp Neo4j
│   ├── retrieval/          # bm25, dense, hybrid, rerank
│   └── eval/               # metric dùng chung
├── experiments/
│   ├── runs/               # mỗi lần chạy: config + metrics.json + log
│   └── reports/            # bảng tổng hợp markdown cho từng bài toán
└── notebooks/              # chỉ để xem kết quả, không chứa logic chính
```

Mỗi lần chạy thí nghiệm ghi vào `experiments/runs/<bai_toan>/<timestamp>_<method>/` gồm: `config.yaml`, `metrics.json`, thời gian chạy, chi phí API (nếu có), version model.

## 4. Các bài toán con và phương pháp cần so sánh

### BT1 — Thu thập dữ liệu

- Lấy metadata + abstract từ Semantic Scholar (bulk search / paper batch) và arXiv; ghép theo arXiv ID / DOI.
- Xử lý: phân trang, rate limit (retry + exponential backoff), cache response vào `data/raw/` để không gọi lại.
- **Đo:** tỉ lệ bài có abstract, tỉ lệ trùng lặp giữa 2 nguồn, tỉ lệ ghép được, thời gian thu thập.
- **So sánh:** (a) chỉ Semantic Scholar, (b) chỉ arXiv, (c) kết hợp — về độ phủ abstract và metadata.

### BT2 — Trích xuất thực thể từ abstract

Loại thực thể: `Task`, `Method`, `Dataset`, `Metric` (có thể thêm `Material/Tool` nếu cần).

Phương pháp cần thử:
1. Baseline từ điển/regex (danh sách dataset/method phổ biến + pattern viết hoa, viết tắt).
2. spaCy NER tổng quát (`en_core_web_sm`) — baseline yếu để đối chiếu.
3. SciSpaCy (`en_core_sci_sm`) + ánh xạ sang loại thực thể.
4. Mô hình đã fine-tune cho NER khoa học trên Hugging Face (dạng SciERC) — tìm và **xác minh tên model trước khi dùng**.
5. LLM zero-shot với output JSON theo schema.
6. LLM few-shot (3–5 ví dụ) với output JSON theo schema.

- **Tập đánh giá:** gán nhãn thủ công 50–100 abstract từ `tiny` (lưu `data/annotations/ner_gold.jsonl`). Viết script hỗ trợ gán nhãn đơn giản (ví dụ: chạy LLM tạo nhãn nháp → người sửa).
- **Metric:** Precision/Recall/F1 theo từng loại thực thể, cả exact match và partial match; thời gian/abstract; chi phí/1.000 abstract.
- **Lưu ý LLM:** cache kết quả theo hash(prompt + abstract); hỏi người dùng trước khi chạy LLM trả phí trên quy mô > `tiny`.

### BT3 — Trích xuất quan hệ

Quan hệ chính: `(Paper)-[:PROPOSES|USES]->(Method)`, `(Paper)-[:EVALUATES_ON]->(Dataset)`, `(Paper)-[:ADDRESSES]->(Task)`, `(Paper)-[:REPORTS]->(Metric)`.

Phương pháp:
1. Heuristic: thực thể xuất hiện trong abstract → gán quan hệ mặc định theo loại.
2. Pattern ngữ pháp/từ khóa ("we propose", "evaluated on", "outperforms").
3. LLM trích xuất thực thể + quan hệ cùng lúc (so với tách 2 bước).

- **Metric:** P/R/F1 trên tập gold (dùng chung tập abstract với BT2).

### BT4 — Chuẩn hóa thực thể (entity normalization)

Ví dụ: "BERT", "Bert", "BERT-base", "bidirectional encoder representations from transformers" → một node.

Phương pháp:
1. Lowercase + bỏ ký tự đặc biệt.
2. Fuzzy matching (`rapidfuzz`) với ngưỡng (thử nhiều ngưỡng).
3. Embedding tên thực thể + clustering (agglomerative với ngưỡng cosine).
4. LLM canonicalization (đưa danh sách ứng viên, yêu cầu gom nhóm).

- **Tập đánh giá:** gán nhãn thủ công ~200 cặp tên thực thể (cùng / khác).
- **Metric:** pairwise Precision/Recall/F1; số node sau chuẩn hóa.

### BT5 — Thiết kế và nạp Knowledge Graph

- Schema v1: `Paper`, `Author`, `Venue`, `FieldOfStudy`, `Task`, `Method`, `Dataset`, `Metric`; quan hệ `AUTHORED`, `PUBLISHED_IN`, `CITES`, `HAS_FIELD` + các quan hệ ở BT3.
- Ghi schema ra `src/kg/schema.md` (node, thuộc tính, quan hệ, constraint/index).
- Nạp bằng `MERGE` theo batch (`UNWIND`), tạo uniqueness constraint trước khi nạp.
- **Đo:** thời gian nạp theo từng mức dữ liệu, số node/edge, và chạy một bộ ~10 truy vấn Cypher mẫu (ví dụ: "các dataset thường dùng cho task X", "phương pháp nào được dùng nhiều nhất trên dataset Y", "bài báo của tác giả Z về task X") — ghi kết quả và thời gian truy vấn.

### BT6 — Tìm kiếm từ khóa (BM25)

Biến thể cần thử:
- Field dùng để index: `title` / `abstract` / `title + abstract` (thử nhân trọng số title).
- Tiền xử lý: lowercase, stopword, stemming vs không stemming.
- Tham số `k1`, `b` (grid nhỏ).
- So sánh `rank_bm25` với một thư viện nhanh hơn (ví dụ `bm25s`) về tốc độ khi lên `medium`.

### BT7 — Tìm kiếm ngữ nghĩa (embedding + FAISS)

Model embedding cần so sánh (xác minh tên trên Hugging Face trước khi dùng):
- `sentence-transformers/all-MiniLM-L6-v2` (baseline nhẹ)
- `allenai/specter2_base` (chuyên cho bài báo khoa học)
- `BAAI/bge-small-en-v1.5` hoặc `bge-base`
- `intfloat/e5-small-v2` (lưu ý prefix `query:` / `passage:`)

Index FAISS: `IndexFlatIP` (chính xác, dùng làm chuẩn) vs `IndexHNSWFlat` vs `IndexIVFFlat` — so sánh recall so với Flat, tốc độ, bộ nhớ (chỉ có ý nghĩa ở `medium`).

### BT8 — Tìm kiếm lai (hybrid) và rerank

1. Reciprocal Rank Fusion (RRF) BM25 + dense.
2. Kết hợp tuyến tính điểm đã chuẩn hóa (thử alpha 0.0 → 1.0, bước 0.1).
3. KG-augmented: nhận diện thực thể trong query → mở rộng ứng viên bằng các bài nối với cùng thực thể trong Neo4j.
4. (Tùy chọn) Rerank top-50 bằng cross-encoder (ví dụ `cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Đánh giá chung cho BT6–BT8

- **Tập đánh giá nội bộ:** 50 query thực tế (dạng người dùng sẽ hỏi: theo chủ đề, theo phương pháp, theo dataset). Với mỗi query, gom top-20 từ tất cả phương pháp (pooling), người gán nhãn mức liên quan 0/1/2. Lưu `data/annotations/retrieval_qrels.jsonl`.
- **Benchmark công khai (sanity check):** chạy thêm trên SciFact và/hoặc SCIDOCS (thuộc BEIR) vì cũng dựa trên abstract.
- **Metric:** nDCG@10, Recall@10, Recall@100, MRR@10, độ trễ trung bình/query, thời gian build index.

## 5. Quy trình cho mỗi bài toán con

1. Cài baseline đơn giản nhất trước, chạy end-to-end trên `tiny`.
2. Tạo/hoàn thiện tập đánh giá (gold).
3. Cài lần lượt các phương pháp còn lại, chạy trên `tiny`, ghi kết quả.
4. Viết `experiments/reports/<bai_toan>.md`: bảng so sánh, nhận xét, 5–10 ví dụ lỗi tiêu biểu (error analysis), đề xuất phương pháp vào vòng sau.
5. **Dừng, báo cáo cho người dùng.**
6. Chạy 2–3 phương pháp tốt nhất trên `small`, cập nhật báo cáo.
7. Chạy phương pháp được chọn trên `medium`, kiểm tra chất lượng và tốc độ.

## 6. Thứ tự thực hiện

1. BT1 (dữ liệu `tiny`) → tạo `papers_tiny.jsonl`
2. BT6 + BT7 trên `tiny` (nhanh, cho kết quả sớm) + tạo tập query đánh giá
3. BT2 → BT3 → BT4 (trích xuất, cần gán nhãn)
4. BT5 (nạp KG từ kết quả tốt nhất)
5. BT8 (hybrid, KG-augmented)
6. Mở rộng lên `small`, rồi `medium`

## 7. Kết quả cuối Giai đoạn 1

- Bộ dữ liệu `medium` sạch (abstract + metadata).
- KG v1 trên Neo4j + schema đã tài liệu hóa.
- Module tìm kiếm với phương pháp được chọn cho từng bước.
- `experiments/reports/SUMMARY.md`: một bảng duy nhất — mỗi bài toán con, phương pháp đã thử, metric chính, phương pháp được chọn và lý do. Đây là nội dung chính cho báo cáo đồ án.

## 8. Quy tắc làm việc cho Claude Code

- Làm từng bài toán con, không viết toàn bộ hệ thống một lần.
- Không tự ý mở rộng quy mô dữ liệu hoặc chạy LLM trả phí ngoài `tiny` — phải hỏi trước.
- Không bịa số liệu; nếu chưa chạy thì ghi "chưa chạy".
- Không lưu API key trong code; đọc từ `.env` (có `.env.example`).
- Có test tối thiểu cho hàm metric (`src/eval/`) vì mọi kết luận phụ thuộc vào chúng.
- Khi tên model/thư viện không chắc chắn tồn tại, kiểm tra trước rồi mới dùng.
