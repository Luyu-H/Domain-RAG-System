import json
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

def extract_pubmed_id(url):
    """Extract PubMed ID from URL."""
    return url.split('/')[-1]

def fetch_pubmed_info(pubmed_id):
    """
    Get the titles and abstracts from PubMed papers.
    """
    # Use PubMed API to get the information of papers
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pubmed_id,
        'retmode': 'xml',
        'rettype': 'abstract'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        soup = BeautifulSoup(response.content, 'xml')
        
        # Extract titles
        title_tag = soup.find('ArticleTitle')
        title = title_tag.get_text() if title_tag else ""
        
        # Extract abstracts
        abstract_texts = soup.find_all('AbstractText')
        if abstract_texts:
            abstract = ' '.join([abs_text.get_text() for abs_text in abstract_texts])
        else:
            abstract = ""
        
        return {
            'id': pubmed_id,
            'link': f'http://www.ncbi.nlm.nih.gov/pubmed/{pubmed_id}',
            'title': title,
            'abstract': abstract
        }
    
    except Exception as e:
        print(f"Error occurred when obtaining the PubMed ID {pubmed_id}: {str(e)}")
        return {
            'id': pubmed_id,
            'link': f'http://www.ncbi.nlm.nih.gov/pubmed/{pubmed_id}',
            'title': "",
            'abstract': ""
        }

def process_bioasq_documents(input_file, output_file):
    """
    Process the documents field in the BioASQ dataset, extract all unique PubMed article information and save it as JSONL.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Collect all unique PubMed article information
    pubmed_urls = set()
    if 'questions' in data:
        for question in data['questions']:
            if 'documents' in question:
                pubmed_urls.update(question['documents'])
    
    print(f"Find {len(pubmed_urls)} unique PubMed articles...")
    
    articles = []
    for i, url in enumerate(tqdm(pubmed_urls, desc="Fetching PubMed articles"), 1):
        pubmed_id = extract_pubmed_id(url)
        
        article_info = fetch_pubmed_info(pubmed_id)
        articles.append(article_info)
        
        if i % 3 == 0:
            time.sleep(1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print(f"\nDone! The information of {len(articles)} articles has been saved to {output_file}.")

def repair_pubmed_corpus(input_file, output_file):
    """
    Repair the PubMed corpus by ensuring all links in the input file appear in the output file,
    and re-fetch articles with missing title or abstract.
    """

    # Load BioASQ input and collect all unique PubMed URLs
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pubmed_urls = set()
    if 'questions' in data:
        for question in data['questions']:
            if 'documents' in question:
                pubmed_urls.update(question['documents'])
    print(f"Collected {len(pubmed_urls)} unique PubMed URLs from input file.")

    # Load existing corpus if available 
    existing_articles = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                existing_articles[item['id']] = item
        print(f"Loaded {len(existing_articles)} existing entries from {output_file}.")
    except FileNotFoundError:
        print(f"No existing file found at {output_file}, starting fresh.")

    # Identify missing articles
    missing_ids = []
    for url in pubmed_urls:
        pubmed_id = extract_pubmed_id(url)
        if pubmed_id not in existing_articles:
            missing_ids.append(pubmed_id)
    print(f"Found {len(missing_ids)} missing articles not in corpus.")

    # Fetch missing articles
    for i, pubmed_id in enumerate(tqdm(missing_ids, desc="Fetching missing articles"), 1):
        info = fetch_pubmed_info(pubmed_id)
        existing_articles[pubmed_id] = info
        if i % 3 == 0:
            time.sleep(1)

    # Re-fetch articles with missing title or abstract
    need_refetch = [
        pid for pid, item in existing_articles.items()
        if item.get('title', '') == '' or item.get('abstract', '') == ''
    ]
    print(f"Found {len(need_refetch)} articles with missing title/abstract, re-fetching...")

    for i, pubmed_id in enumerate(tqdm(need_refetch, desc="Re-fetching incomplete articles"), 1):
        new_info = fetch_pubmed_info(pubmed_id)
        existing_articles[pubmed_id] = new_info
        if i % 3 == 0:
            time.sleep(1)

    # Save updated corpus
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in existing_articles.values():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nRepair completed. Total {len(existing_articles)} articles saved to {output_file}.")


if __name__ == "__main__":
    input_file = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/bioasq_data_cleaned.json"
    output_file = "/Users/lorraine/Documents/courses/25-26fall/cse291a_yiying_LLM_agent/project/Domain-RAG-System/data/BioASQ/corpus_pubmed.jsonl"
    
    # process_bioasq_documents(input_file, output_file)
    repair_pubmed_corpus(input_file, output_file)