import json

def extract_pubmed_id(url):
    """Extract PubMed ID from URL."""
    return url.split('/')[-1]

def clean_bioasq_data(input_file, output_file):
    """
    1. Remove 'concepts' and 'triples' from the BioASQ task 13b dataset.
    2. Unify the section name (can only be 'abstract' or 'title')
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'questions' in data:
        for question in data['questions']:
            if 'concepts' in question:
                del question['concepts']

            if 'triples' in question:
                del question['triples']

            if 'snippets' in question:
                for snippet in question['snippets']:
                    if 'beginSection' in snippet:
                        snippet['beginSection'] = snippet['beginSection'].replace('sections.0', 'abstract')
                    if 'endSection' in snippet:
                        snippet['endSection'] = snippet['endSection'].replace('sections.0', 'abstract')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Finish cleaning data!")

def clean_invalid_pubmed_entries(input_json_path, corpus_path):
    """
    Clean invalid PubMed entries and remove their references from the BioASQ dataset.
    """

    # Load corpus and find invalid entries
    valid_articles = []
    invalid_ids = set()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("title", "").strip() == "" or item.get("abstract", "").strip() == "":
                invalid_ids.add(item["id"])
            else:
                valid_articles.append(item)

    print(f"Found {len(invalid_ids)} invalid PubMed articles to remove.")

    # Rewrite corpus without invalid entries
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for item in valid_articles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Updated corpus file saved: {len(valid_articles)} valid entries remain.")

    # Load BioASQ dataset and remove references to invalid articles
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    removed_doc_count = 0
    removed_question_count = 0

    if 'questions' in data:
        new_questions = []
        for q in data['questions']:
            if 'documents' in q:
                # Filter out invalid document links
                original_len = len(q['documents'])
                q['documents'] = [
                    doc for doc in q['documents']
                    if extract_pubmed_id(doc) not in invalid_ids
                ]
                removed_doc_count += (original_len - len(q['documents']))

                # Optionally skip question if it now has no documents
                if len(q['documents']) > 0:
                    new_questions.append(q)
                else:
                    removed_question_count += 1
            else:
                new_questions.append(q)
        data['questions'] = new_questions

    # Rewrite BioASQ dataset
    with open(input_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Removed {removed_doc_count} invalid document links "
          f"and {removed_question_count} related questions.")
    print("Both files have been updated successfully.")

if __name__ == "__main__":
    input_path = "./data/BioASQ-training13b/training13b.json"
    output_path = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/bioasq_data_cleaned.json"
    corpus_path = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/corpus_pubmed.jsonl"

    clean_bioasq_data(input_path, output_path)
    clean_invalid_pubmed_entries(output_path, corpus_path)


