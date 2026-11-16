import json
import re
from pathlib import Path

def extract_entities(abstract):
    text = abstract.lower()
    
    # Định nghĩa các từ khóa mẫu phổ biến trong paper AI/CS để nhận diện thực thể
    methods = ["transformer", "scibert", "bert", "gnn", "graph neural network", "cnn", "lstm", "diffusion model", "llm", "large language model", "framework", "approach", "algorithm"]
    datasets = ["cora", "pubmed", "citeseer", "arxiv", "mnist", "imagenet", "glue", "SQuAD", "imdb"]
    metrics = ["f1-score", "accuracy", "bleu", "recall", "precision", "perplexity", "auc", "roc"]
    tasks = ["entity extraction", "relation extraction", "paper search", "classification", "translation", "summarization", "clustering"]
    
    found_methods = [m.title() for m in methods if m in text]
    found_datasets = [d.upper() if len(d) <= 5 else d.title() for d in datasets if d in text]
    found_metrics = [me.upper() if "f1" in me or "auc" in me else me.title() for me in metrics if me in text]
    found_tasks = [t.title() for t in tasks if t in text]
    
    labels = {}
    if found_methods:
        labels["Method"] = list(set(found_methods))
    if found_datasets:
        labels["Dataset"] = list(set(found_datasets))
    if found_metrics:
        labels["Metric"] = list(set(found_metrics))
    if found_tasks:
        labels["Task"] = list(set(found_tasks))
        
    return labels

def main():
    input_path = Path("data/processed/sampled_abstracts_for_labeling.jsonl")
    output_path = Path("data/processed/sampled_abstracts_for_labeling.jsonl") # Ghi đè file đã gán nhãn hoàn chỉnh
    
    if not input_path.exists():
        print(f"Không tìm thấy file {input_path}. Hãy chạy script sample_abstracts.py trước nhé!")
        return

    updated_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                abstract = record.get("abstract", "")
                
                # Tự động gán nhãn dựa trên nội dung abstract
                record["labels"] = extract_entities(abstract)
                updated_records.append(record)

    # Ghi lại vào file jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã tự động gán nhãn thành công cho {len(updated_records)} bài báo vào file: {output_path}")

if __name__ == "__main__":
    main()