import json
import random

def extract_pubmed_id(url):
    """Extract PubMed ID from URL."""
    return url.split('/')[-1]

def sample_bioasq_and_corpus(
    input_json_path,
    corpus_path,
    output_json_path="bioasq_subset.json",
    output_corpus_path="corpus_pubmed_subset.jsonl",
    sample_size=50,
    min_corpus_size=3000,
    seed=42
):
    """
    Randomly sample BioASQ questions by type and corresponding PubMed documents.
    If the number of related corpus entries is less than `min_corpus_size`,
    randomly add unrelated corpus entries to reach the desired number.

    Args:
        input_json_path (str): Path to bioasq_data_cleaned.json
        corpus_path (str): Path to corpus_pubmed.jsonl
        output_json_path (str): Output path for sampled BioASQ subset
        output_corpus_path (str): Output path for corresponding PubMed corpus
        sample_size (int): Number of questions per type to sample
        min_corpus_size (int): Minimum number of corpus documents required
        seed (int): Random seed for reproducibility
    """

    random.seed(seed)

    # Load BioASQ dataset
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"Loaded {len(questions)} total questions from {input_json_path}")

    # Group questions by type
    type_groups = {}
    for q in questions:
        q_type = q.get("type", "unknown")
        type_groups.setdefault(q_type, []).append(q)
    print(f"Found {len(type_groups)} question types: {list(type_groups.keys())}")

    # Randomly sample questions per type
    sampled_questions = []
    for q_type, q_list in type_groups.items():
        n = min(sample_size, len(q_list))
        sampled = random.sample(q_list, n)
        sampled_questions.extend(sampled)
        print(f"Sampled {n} questions from type '{q_type}'")

    # Collect all referenced PubMed IDs
    sampled_pubmed_ids = set()
    for q in sampled_questions:
        for doc_url in q.get("documents", []):
            sampled_pubmed_ids.add(extract_pubmed_id(doc_url))
    print(f"Collected {len(sampled_pubmed_ids)} unique PubMed IDs from sampled questions.")

    # Load corpus and select corresponding documents
    all_corpus = []
    selected_corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            all_corpus.append(item)
            if item["id"] in sampled_pubmed_ids:
                selected_corpus.append(item)

    print(f"Selected {len(selected_corpus)} matching articles from corpus.")

    # If corpus is too small, randomly add unrelated documents
    if len(selected_corpus) < min_corpus_size:
        needed = min_corpus_size - len(selected_corpus)
        print(f"Corpus size {len(selected_corpus)} < {min_corpus_size}, adding {needed} random docs.")
        remaining = [art for art in all_corpus if art["id"] not in sampled_pubmed_ids]
        add_count = min(needed, len(remaining))
        selected_corpus.extend(random.sample(remaining, add_count))
        print(f"Final corpus size: {len(selected_corpus)}")

    # Save subset BioASQ file
    subset_data = {"questions": sampled_questions}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, ensure_ascii=False, indent=2)
    print(f"Saved sampled BioASQ data to {output_json_path}")

    # Save subset corpus file
    with open(output_corpus_path, 'w', encoding='utf-8') as f:
        for item in selected_corpus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved sampled corpus to {output_corpus_path}")

    print("Sampling completed successfully!")


# Example usage
if __name__ == "__main__":
    input_json_path = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/bioasq_data_cleaned.json"
    corpus_path = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/corpus_pubmed.jsonl"
    sample_bioasq_and_corpus(
        input_json_path,
        corpus_path,
        output_json_path="bioasq_subset.json",
        output_corpus_path="corpus_subset.jsonl",
        sample_size=50
    )
