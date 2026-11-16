import json
import random
from pathlib import Path

def main():
    input_path = Path("data/processed/papers_tiny.jsonl")
    output_path = Path("data/processed/sampled_abstracts_for_labeling.jsonl")
    
    if not input_path.exists():
        print(f"Không tìm thấy file {input_path}. Hãy chạy bt1_collect.py trước nhé!")
        return

    # Đọc tất cả các dòng từ file tiny jsonl
    papers = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))

    # Lấy mẫu ngẫu nhiên 80 bài (nằm trong khoảng 50-100)
    sample_size = min(80, len(papers))
    sampled_papers = random.sample(papers, sample_size)

    # Ghi ra file mới để gán nhãn
    with open(output_path, "w", encoding="utf-8") as f:
        for p in sampled_papers:
            # Lấy id theo các tên key phổ biến trong dự án để tránh bị null
            paper_id = p.get("id") or p.get("paper_id") or p.get("sha") or p.get("arxiv_id")
            
            # Chỉ lưu các trường cần thiết cho việc gán nhãn (id, title, abstract)
            record = {
                "id": paper_id,
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "labels": {} # Chỗ để điền nhãn entity sau khi gán
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã trích xuất thành công {sample_size} abstract vào file: {output_path}")

if __name__ == "__main__":
    main()