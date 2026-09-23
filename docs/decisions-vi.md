# 6 quyết định cần chốt

**Gửi:** Hà My · **Từ:** Minh Hiếu · **Ngày:** 23/09/2026

Mình vừa dựng xong bộ khung repo và viết bản phân tích yêu cầu. Trong quá trình đó có 6
chỗ mình không tự quyết được vì nó ảnh hưởng tới cả hai phần việc, nên cần My đọc và cho ý
kiến.

Tài liệu này viết để đọc độc lập — không cần đọc `requirements.md` trước. Mỗi mục mình ghi:
vấn đề là gì, có mấy cách làm, mình nghiêng về cách nào và vì sao. My chỉ cần trả lời
"đồng ý" hoặc nói rõ chỗ không đồng ý.

Bản phân tích đầy đủ (tiếng Anh, có số liệu chi tiết và các phương án bị loại) nằm ở
[`decisions.md`](decisions.md) — file đó là bản ghi chính thức, file này chỉ để trao đổi.

---

## Đọc cái nào trước

Không phải cả 6 cái đều cần My quyết. Bảng này để My biết chỗ nào là phần của mình:

| # | Quyết định | Chủ yếu ai quyết | Chặn việc gì | Hạn |
|---|---|---|---|---|
| 1 | Có mở rộng dữ liệu qua references không | **My** | Thu thập dữ liệu | Đầu tháng 10 |
| 4 | Dùng LLM nào để trích xuất thực thể | **My** | Trích xuất thông tin | Tháng 11 |
| 6 | Quy ước branch / commit / review | Cả hai | Ngay khi cả hai cùng push code | **Tuần này** |
| 2 | Cắt văn bản thành chunk kiểu gì | Hiếu | Xây index tìm kiếm | Tháng 11 |
| 5 | Chọn model embedding | Hiếu | Xây index tìm kiếm | Tháng 11 |
| 3 | Cách chấm độ trung thực của câu trả lời | Cả hai | Đánh giá Phase 2 | Học kỳ sau |

Gấp nhất là **#6** (10 phút, chốt xong là từ mai làm việc theo quy trình) và **#1** (tháng
10 đã phải thu dữ liệu rồi). Mục 2, 3, 5 để sau cũng được, nhưng My đọc lướt qua thì tốt.

Ngoài ra có **một việc không phải quyết định nhưng cần làm tuần này** — xem
[cuối tài liệu](#việc-cần-làm-ngay-không-phải-quyết-định).

---

## 1. Có mở rộng dữ liệu qua references không?

**Phần của My · chặn việc thu thập dữ liệu tháng 10**

### Vấn đề

Kế hoạch ban đầu: thu 1.000 bài về chủ đề RAG, dựng đồ thị tri thức với quan hệ `CITES`
(bài A trích dẫn bài B). Mình đã viết script thử và đo thật. Kết quả không như mong đợi:

Lấy mẫu 100 bài, xem danh sách tài liệu tham khảo của từng bài:

- Tổng cộng **2.319 lượt trích dẫn**, trỏ tới **2.022 bài khác nhau**
- Nhưng **chỉ 12 lượt** là trỏ tới bài cũng nằm trong tập của mình

Tức là 99,5% trích dẫn trỏ ra ngoài. Quy đổi ra tập 1.000 bài thì mỗi bài chỉ có khoảng
**1,2 quan hệ `CITES` nội bộ**.

Lý do: chủ đề RAG quá mới. Trong 1.000 bài thì 2026 có 400 bài, 2025 có 399 bài — tức là
gần 80% ra đời trong 2 năm gần đây. Bài mới thì chưa ai kịp trích dẫn, mà nó trích dẫn ra
ngoài thì lại trỏ tới bài không nằm trong tập.

Hệ quả: chức năng "bài nào trích dẫn bài này", "mở rộng kết quả tìm kiếm theo mạng trích
dẫn" gần như vô dụng. Mà đó là một phần lý do mình chọn dùng đồ thị thay vì database
thường — nếu bỏ luôn thì phần biện luận trong báo cáo sẽ yếu.

### Có 3 cách

**Cách A — giữ nguyên 1.000 bài.** Chấp nhận đồ thị trích dẫn thưa, tập trung nối bài qua
`USES_METHOD` / `USES_DATASET` / `HAS_TOPIC` thay vì `CITES`.

**Cách B — thêm các bài được trích dẫn nhiều.** Lấy danh sách references của 1.000 bài, bài
nào được từ 3 bài trở lên trích dẫn thì thêm vào đồ thị. Ước tính thêm khoảng 1.500–3.000
bài. Đây là các bài nền tảng của lĩnh vực (kiểu BERT, DPR, bài RAG gốc) — ai cũng trích dẫn.

**Cách C — thêm tất cả.** Khoảng 15.000–18.000 bài. Phần lớn là bài chỉ được trích dẫn đúng
một lần.

### Chỗ mình nghĩ My sẽ quan tâm nhất

Lúc đầu mình tưởng cách B sẽ đắt — thêm mấy nghìn bài nghĩa là thêm mấy nghìn PDF phải tải,
phải trích xuất thực thể, phải embed. **Nhưng không phải.**

Các bài thêm vào chỉ tồn tại để làm *đích đến* của quan hệ trích dẫn. Chúng **không cần**
tải PDF, **không cần** chạy LLM trích xuất, **không cần** embedding. Những bước tốn kém đó
vẫn chỉ chạy trên 1.000 bài gốc.

Chi phí thật của cách B so với cách A: **bằng nhau**. Vì kể cả cách A cũng phải gọi API lấy
danh sách references (đó là cách duy nhất tạo ra 1.200 cạnh nội bộ). Cách B chỉ là "đã lấy
về rồi thì lưu luôn các bài đó lại làm node".

### Mình đề xuất cách B

Ngưỡng ≥3 bài trích dẫn. Con số này mình ước từ mẫu 100 bài, khi chạy thật trên 1.000 bài
thì đo lại rồi chỉnh.

**Nếu chọn B thì schema đồ thị phải sửa một chỗ** — cái này liên quan trực tiếp tới
`kg-schema.md` My sắp viết:

Node `Paper` sẽ có hai loại. Bài gốc thì tìm kiếm được (có abstract, có full text, có trong
index). Bài thêm vào thì chỉ có metadata, không tìm kiếm được. Nếu không phân biệt, hệ
thống sẽ trả về một bài mà người dùng bấm vào thì không có gì để đọc — trông như lỗi.

Mình đề xuất thêm thuộc tính `in_corpus: true/false` trên node `Paper`, hoặc gắn nhãn thứ
hai `:External`. My thấy cách nào hợp với schema hơn thì chọn.

> **My trả lời:** ☐ cách A ☐ cách B ☐ cách C · Nếu B: đánh dấu bằng `in_corpus` hay `:External`?

---

## 2. Cắt văn bản thành chunk kiểu gì?

**Phần của Hiếu · My đọc để biết, vì nó ảnh hưởng tới phần provenance**

### Vấn đề

Để tìm kiếm ngữ nghĩa, phải cắt bài báo thành từng đoạn nhỏ rồi embed từng đoạn. Cắt to hay
nhỏ ảnh hưởng hai thứ ngược chiều nhau: trích dẫn chỉ được chính xác tới đơn vị mình cắt,
nhưng model embedding có giới hạn đầu vào (thường 512 token) — đoạn dài hơn thì **phần
đuôi bị cắt bỏ âm thầm**, không báo lỗi gì cả.

### Có 3 cách

| | A — Cắt theo section | B — Cắt theo cửa sổ cố định | C — Kết hợp |
|---|---|---|---|
| Đơn vị | mỗi section 1 chunk | 512 token, chồng lấn 64 | cắt theo section trước, section nào dài quá thì cắt tiếp |
| Trích dẫn hiện ra | "phần Method của [12]" | "đoạn 47 của [12]" | "phần Method của [12]" |
| Chất lượng embedding | kém với section dài — bị cắt đuôi | đều | đều |
| Rủi ro | section Experiments 3.000 từ thì retrieval gần như không thấy | cắt ngang giữa 2 section, trộn 2 chủ đề vào 1 vector | code nhiều hơn chút |

### Mình đề xuất cách C

Vì nó là cách duy nhất vừa giữ được trích dẫn đọc ra có nghĩa, vừa không vượt giới hạn
model.

**Chỗ liên quan tới My:** mỗi chunk phải mang theo thông tin nguồn gốc, và phần này phải
khớp với cách My tách section trong bước xử lý PDF:

```
paper_id · nguồn ("abstract" hoặc tên section) · số thứ tự cửa sổ trong nguồn đó · vị trí ký tự
```

Nhớ là khoảng 360/1.000 bài không có full text — với những bài đó thì abstract chính là
chunk duy nhất, và phải ghi rõ `nguồn = "abstract"` để lúc trả lời còn biết mà nói "cái này
chỉ dựa trên tóm tắt thôi".

Nên chuẩn tên section My dùng lúc tách PDF sẽ trở thành một phần của định danh chunk. Hai
đứa cần thống nhất danh sách tên section chuẩn (Abstract, Introduction, Related Work,
Method, Experiments, Results, Conclusion — bài nào có mục tên khác thì map vào đâu).

> **My trả lời:** ☐ đồng ý cách C · Danh sách tên section chuẩn: ______

---

## 3. Chấm độ trung thực của câu trả lời thế nào?

**Cả hai · học kỳ sau mới cần, nhưng nên chốt cách chấm trước khi thấy kết quả**

### Vấn đề

Trong yêu cầu mình đặt mục tiêu "≥90% khẳng định trong câu trả lời phải có căn cứ từ tài
liệu truy xuất được". Con số này vô nghĩa nếu chưa định rõ: một "khẳng định" là gì, chấm
bao nhiêu mẫu, ai chấm, hai người chấm lệch nhau thì xử sao.

Quan trọng: phải chốt **trước** khi nhìn thấy kết quả. Chốt sau thì rất dễ vô tình chỉnh
tiêu chí cho vừa với kết quả mình có, và hội đồng sẽ hỏi đúng chỗ đó.

### Mình đề xuất

| Điểm | Đề xuất | Vì sao |
|---|---|---|
| Đơn vị chấm | Một khẳng định đơn — một ý có thể đối chiếu với một nguồn | Chấm cả câu trả lời thì che mất lỗi cục bộ |
| Cỡ mẫu | 50 câu trả lời, mỗi câu ~4 khẳng định ≈ 200 khẳng định | Đủ để sai số khoảng ±7%, mà hai đứa chấm một buổi tối là xong |
| Thang | đúng / đúng một phần / sai | Thang nhị phân giấu mất trường hợp ở giữa, mà đó mới là chỗ đáng bàn |
| Ai chấm | **Cả hai chấm cùng một mẫu, độc lập, không nhìn kết quả của nhau** | Một người chấm thì đó là ý kiến, không phải đo lường |
| Độ đồng thuận | Báo cáo thêm chỉ số Cohen's kappa | Biến "bọn em có kiểm tra" thành con số bảo vệ được. Cũng là một ý hay để viết vào luận văn |
| Lệch nhau | Ngồi thống nhất lại, lấy nhãn sau khi thống nhất | |

Cách này dùng luôn cho phần kiểm tra độ chính xác trích dẫn: với mỗi khẳng định có trích
dẫn, kiểm tra bài đó có thật không và section được trích có thật chứa nội dung đó không.

> **My trả lời:** ☐ đồng ý ☐ sửa: ______

---

## 4. Dùng LLM nào để trích xuất thực thể?

**Phần của My · chặn bước trích xuất tháng 11**

### Vấn đề

Ban đầu mình ghi đây là một quyết định, nhưng thực ra là **hai việc khác nhau, câu trả lời
cũng khác nhau**:

- **Trích xuất thực thể** (phần của My): chạy một lần trên 1.000 bài, không cần nhanh, cần
  trả về JSON đúng cấu trúc để còn chuẩn hoá tên thực thể.
- **Sinh câu trả lời** (phần của Hiếu, học kỳ sau): chạy lặp đi lặp lại, cần nhanh.

### Về phần trích xuất — chi phí không phải vấn đề như mình tưởng

Mình tính thử: 1.000 bài, mỗi bài khoảng 900 token đầu vào (prompt + tiêu đề + abstract) và
300 token đầu ra. Tổng ~0,9 triệu token vào, ~0,3 triệu token ra.

| Model | Giá vào / ra (mỗi 1 triệu token) | Tổng | Qua Batch API (giảm 50%) |
|---|---|---|---|
| Claude Haiku 4.5 | $1 / $5 | ~$2,40 | **~$1,20** |
| Claude Sonnet 5 | $2 / $10 | ~$4,80 | ~$2,40 |
| Claude Opus 5 | $5 / $25 | ~$12,00 | ~$6,00 |

*(Giá niêm yết tháng 6/2026, cần kiểm lại trước khi dùng thật.)*

Lý do duy nhất để chạy model local là tiết kiệm tiền — mà ở đây chỉ tiết kiệm được **1 đến
6 đô**. Đổi lại: JSON trả về kém ổn định hơn (làm bước chuẩn hoá tên thực thể của My khó
hơn hẳn), và mất vài tiếng chạy CPU laptop.

Trần ngân sách cả dự án mình đặt là 50 đô. Trích xuất tiêu hết ~1-6 đô, không đáng lo.

### Mình đề xuất

1. **Trích xuất: dùng `claude-haiku-4-5` qua Batch API** (~$1,20). Trích xuất là việc hẹp,
   định nghĩa rõ ràng — đúng chỗ model rẻ nhất làm tốt.
2. **Thử 20 bài trước khi chạy cả nghìn.** My xem JSON trả về bằng mắt. Nếu Haiku không ổn
   thì nâng lên `claude-sonnet-5` — chênh 2 đô, không phải chuyện ngân sách.
3. **Phần sinh câu trả lời để đầu học kỳ sau quyết.** Cái đó mới là chỗ tốn: khoảng 2.000
   lượt hỏi lúc phát triển và đánh giá tốn ~$13 với Haiku, ~$26 với Sonnet — gần hết trần.
4. **Ghi lại chi tiêu từ lần gọi API đầu tiên.** Một file ghi chung, cộng dồn. Trần 50 đô
   rất dễ vượt mà không để ý.

> **My trả lời:** ☐ đồng ý Haiku + Batch ☐ khác: ______

---

## 5. Chọn model embedding nào?

**Phần của Hiếu · để My biết là có ràng buộc gì**

### Vấn đề

Ước tính phải embed khoảng 20.000 chunk (640 bài có full text × ~30 chunk, cộng 1.000
abstract), chạy trên CPU laptop, không có GPU.

### Ứng viên

Đều có sẵn trong `sentence-transformers`, thư viện đã nằm trong `requirements.txt`.

| Model | Số chiều | Kích thước | Ghi chú |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~22M | Nhanh nhất, chất lượng yếu nhất. 20k chunk chạy vài phút |
| `BAAI/bge-small-en-v1.5` | 384 | ~33M | Tốt hơn MiniLM rõ rệt, tốc độ gần bằng |
| `all-mpnet-base-v2` | 768 | ~110M | Mạnh, chậm gấp ~4, index to gấp đôi |
| `BAAI/bge-base-en-v1.5` | 768 | ~109M | Tốt nhất trong 4 cái, chi phí như mpnet |

### Mình đề xuất `BAAI/bge-small-en-v1.5`

Đổi lại chất lượng trên mỗi giây CPU là tốt nhất, 384 chiều giữ index FAISS nhỏ. Hai điều
kiện kèm theo:

- **Đo chứ không tin suông.** Khi có bộ câu hỏi kiểm thử (tháng 12), chạy thử 3 model trên
  cùng bộ đó rồi so. Cái này không phải việc thừa — kế hoạch tháng 12 vốn đã yêu cầu đo
  baseline, nên nó chính là một kết quả để đưa vào báo cáo.
- **Ghi cứng tên model vào `config.py`.** Embedding lúc dựng index và lúc tìm kiếm phải cùng
  một model; đổi model là phải embed lại từ đầu.

> **My trả lời:** ☐ đồng ý ☐ ý kiến khác: ______

---

## 6. Quy ước làm việc chung

**Cả hai · cần chốt tuần này**

### Vấn đề

Hai người, một repo, chưa thống nhất quy ước gì. Đây là quyết định rẻ nhất và gấp nhất —
code nào push trước khi chốt thì sau phải dọn lại. Bản kế hoạch cũng liệt kê "review pull
request giữa hai thành viên" là một hạng mục phải nộp, nên đây là việc bắt buộc chứ không
phải tuỳ chọn.

### Mình đề xuất

**Tên branch** — `<loại>/<mô-tả-ngắn>`:

```
feat/semantic-scholar-client      fix/pdf-hyphenation
docs/kg-schema                    chore/pin-embedding-model
```

**Commit message** — câu mô tả ở thể mệnh lệnh, tối đa 72 ký tự, không có dấu chấm cuối.
Phần thân giải thích **tại sao** làm, không cần kể lại đã sửa gì (nhìn diff là thấy). Viết
tiếng Anh cho khớp với các commit đã có.

**Pull request** — từ giờ không push thẳng lên `main` nữa. Mỗi PR người kia review, một
approve là merge được. Ngoài chuyện đây là hạng mục phải nộp, đọc code của nhau cũng là
cách để cả hai vẫn hiểu được nửa còn lại của hệ thống — lúc viết báo cáo và bảo vệ thì cần.

**Ai phụ trách thư mục nào** — để tránh hai đứa sửa cùng một file:

| Mảng | Phụ trách | Thư mục |
|---|---|---|
| Thu thập, xử lý PDF, đồ thị tri thức | **Hà My** | `src/paper_search/collection/`, `extraction/`, `graph/` |
| Tìm kiếm, RAG, giao diện | **Minh Hiếu** | `src/paper_search/indexing/`, `retrieval/`, `rag/`, `app/` |
| Dùng chung | Cả hai | `config.py`, `docs/`, `tests/`, `pyproject.toml` |

"Phụ trách" nghĩa là review mọi thay đổi trong đó, không phải là người khác không được sửa.

**Trước khi push** — chạy `make lint && make test`. Hoặc cài hook cho nó tự chạy:

```bash
.venv/bin/pre-commit install
```

**Không commit dữ liệu.** File `.gitignore` đã chặn sẵn `data/`, `.env` và các file index.
Cần chia sẻ dữ liệu thì chia sẻ script thu thập kèm số phiên bản dữ liệu, không đẩy file
kết quả lên.

> **My trả lời:** ☐ đồng ý ☐ sửa: ______

---

## Việc cần làm ngay (không phải quyết định)

**Đăng ký API key của Semantic Scholar** — <https://www.semanticscholar.org/product/api>

Lúc chạy script thử, mình dùng chung hạn mức miễn phí không cần key. Endpoint tìm kiếm thì
ổn — lấy 1.000 bài trong đúng một lần gọi. Nhưng endpoint lấy danh sách references (chính
là cái cần cho quyết định #1) trả lỗi 429 "quá nhiều request" **chín lần liên tiếp**, mỗi
lần chờ 25 giây, rồi mới qua được một lần.

Tháng 10 sẽ cần gọi endpoint đó hàng chục lần. Duyệt key không phải lúc nào cũng nhanh, nên
việc này nằm trên đường găng của quyết định #1 dù bản thân nó không phải quyết định.

Có key rồi thì bỏ vào file `.env`, dòng `SEMANTIC_SCHOLAR_API_KEY=` — phần code đọc key đã
viết sẵn rồi.

---

## Trả lời thế nào cũng được

Nhắn trực tiếp, hoặc sửa thẳng vào file này rồi tạo pull request — coi như tập luôn quy ước
ở mục 6.

Chốt xong cái nào mình ghi vào [`decisions.md`](decisions.md), file đó là bản ghi chính
thức để nộp kèm báo cáo Phase 1.

---

## My phản hồi tổng hợp 

- **Quyết định 1:** Chọn **Cách B** (thêm các bài được $\ge 3$ bài trích dẫn). Đánh dấu bằng nhãn thứ hai `:External` trên node `Paper`.
- **Quyết định 2:** Đồng ý **Cách C**. Bộ section chuẩn gồm: `Abstract`, `Introduction`, `Related Work`, `Method`, `Experiments`, `Results`, `Conclusion`, `Other`.
- **Quyết định 3:** **Đồng ý** cách chấm chéo 50 câu độc lập. Bổ sung dùng *Weighted Cohen's Kappa* và tính điểm: Đúng = 1, Đúng một phần = 0.5, Sai = 0.
- **Quyết định 4:** **Đồng ý** dùng `claude-haiku-4-5` qua Batch API. Sau khi My kiểm tra 20 bài đầu, nếu tên thực thể bị phân mảnh thì nâng lên `claude-sonnet-5`.
- **Quyết định 5:** **Đồng ý** chọn `BAAI/bge-small-en-v1.5`. Lưu ý nhớ thêm instruction khi embed query tìm kiếm.
- **Quyết định 6:** **Đồng ý** toàn bộ quy ước Git, phân chia thư mục và pre-commit.
- **Việc cần làm ngay:** My đang đăng ký API key Semantic Scholar và sẽ thêm vào file `.env` sớm.

