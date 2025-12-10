import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    source: str  # 'pubmed', 'openfda', 'kaggle'
    title: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            'doc_id': self.doc_id,
            'source': self.source,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata
        }


class DataLoader:
    """Load and preprocess data from multiple sources (PubMed, Kaggle, OpenFDA)."""
    
    def __init__(self, pubmed_path: str, openfda_path: str, kaggle_path: str):
        self.pubmed_path = Path(pubmed_path)
        self.openfda_path = Path(openfda_path)
        self.kaggle_path = Path(kaggle_path)
        
    def load_all(self) -> List[Document]:
        documents = []

        pubmed_path = self.pubmed_path
        if pubmed_path.exists():
            documents.extend(self.load_pubmed(pubmed_path))

        openfda_path = self.openfda_path
        if openfda_path.exists():
            documents.extend(self.load_openfda(openfda_path))
            
        kaggle_path = self.kaggle_path
        if kaggle_path.exists():
            documents.extend(self.load_kaggle(kaggle_path))
            
        print(f"Finish loading {len(documents)} docs")
        return documents
    
    def load_pubmed(self, filepath: Path) -> List[Document]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                doc_id=f"pubmed_{item['id']}",
                source='pubmed',
                title=item.get('title', ''),
                content=item.get('abstract', ''),
                metadata={
                    'link': item.get('link', ''),
                    'pubmed_id': item['id']
                }
            )
            documents.append(doc)
        
        print(f"PubMed: {len(documents)} docs")
        return documents
    
    def load_openfda(self, filepath: Path) -> List[Document]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            title = item.get('brand_name', '')
            if item.get('generic_name'):
                title = f"{title} ({item['generic_name']})"
            
            doc = Document(
                doc_id=f"openfda_{item['id']}",
                source='openfda',
                title=title,
                content=item.get('text', ''),
                metadata={
                    'fda_id': item['id'],
                    'set_id': item.get('set_id', ''),
                    'brand_name': item.get('brand_name', ''),
                    'generic_name': item.get('generic_name', ''),
                    'effective_time': item.get('metadata', {}).get('effective_time', ''),
                    'version': item.get('metadata', {}).get('version', ''),
                    'has_active_ingredients': item.get('metadata', {}).get('has_active_ingredients', False),
                    'has_warnings': item.get('metadata', {}).get('has_warnings', False),
                    'has_dosage': item.get('metadata', {}).get('has_dosage', False),
                    'has_indications': item.get('metadata', {}).get('has_indications', False),
                }
            )
            documents.append(doc)
        
        print(f"OpenFDA: {len(documents)} docs")
        return documents
    
    def load_kaggle(self, filepath: Path) -> List[Document]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            content = self._build_kaggle_content(item)
            
            doc = Document(
                doc_id=f"kaggle_{item['doc_id']}",
                source='kaggle',
                title=f"{item['drug_name']} for {item['medical_condition']}",
                content=content,
                metadata={
                    'drug_name': item['drug_name'],
                    'generic_name': item.get('generic_name', ''),
                    'medical_condition': item['medical_condition'],
                    'drug_classes': item.get('drug_classes', []),
                    'brand_names': item.get('brand_names', []),
                    'rx_otc': item.get('rx_otc', ''),
                    'pregnancy_category': item.get('pregnancy_category', ''),
                    'pregnancy_category_raw': item.get('pregnancy_category_raw', ''),
                    'csa': item.get('csa', ''),
                    'alcohol': item.get('alcohol', ''),
                    'rating': item.get('rating'),
                    'activity': item.get('activity'),
                    'no_of_reviews': item.get('no_of_reviews'),
                    'links': item.get('links', {}),
                    'has_side_effects': bool(item.get('side_effects_structured')),
                    'has_condition_summary': bool(item.get('condition_summary'))
                }
            )
            documents.append(doc)
        
        print(f"Kaggle Drug Data: {len(documents)} docs")
        return documents
    
    def _build_kaggle_content(self, item: Dict) -> str:
        sections = []
    
        # 1. basic information
        basic_info = []
        basic_info.append(f"Drug Name: {item['drug_name']}")
        
        if item.get('generic_name') and item['generic_name'] != item['drug_name']:
            basic_info.append(f"Generic Name: {item['generic_name']}")
        
        basic_info.append(f"Medical Condition: {item['medical_condition']}")
        
        # classes of drugs
        if item.get('drug_classes'):
            classes = ', '.join(item['drug_classes'])
            basic_info.append(f"Drug Classes: {classes}")
        
        # prescription type
        if item.get('rx_otc'):
            basic_info.append(f"Prescription Type: {item['rx_otc']}")
        
        # rating, activity, reviews
        if item.get('rating'):
            basic_info.append(f"User Rating: {item['rating']}/10")
        if item.get('activity'):
            basic_info.append(f"Activity Score: {item['activity']}")
        if item.get('no_of_reviews'):
            basic_info.append(f"Number of Reviews: {int(item['no_of_reviews'])}")
        
        sections.append("=== BASIC INFORMATION ===\n" + '\n'.join(basic_info))
        
        # 2. brand names
        if item.get('brand_names'):
            brand_section = []
            brand_section.append("=== BRAND NAMES ===")
            brand_section.append(', '.join(item['brand_names']))
            sections.append('\n'.join(brand_section))
        
        # 3. side effects (complete retention)
        if item.get('side_effects_structured'):
            se = item['side_effects_structured']
            side_effects_section = []
            side_effects_section.append("=== SIDE EFFECTS ===")
            
            # serious side effects
            if se.get('serious'):
                side_effects_section.append("\nSerious Side Effects:")
                for i, effect in enumerate(se['serious'], 1):
                    side_effects_section.append(f"  {i}. {effect}")
            
            # common side effects
            if se.get('common'):
                side_effects_section.append("\nCommon Side Effects:")
                for i, effect in enumerate(se['common'], 1):
                    side_effects_section.append(f"  {i}. {effect}")
            
            sections.append('\n'.join(side_effects_section))
        
        if item.get('side_effects_raw'):
            sections.append(f"=== SIDE EFFECTS (RAW) ===\n{item['side_effects_raw']}")
        
        # 4. safety information
        safety_info = []
        if item.get('pregnancy_category') or item.get('alcohol') or item.get('csa'):
            safety_info.append("=== SAFETY INFORMATION ===")
            
            if item.get('pregnancy_category'):
                preg_text = item['pregnancy_category']
                if item.get('pregnancy_category_raw'):
                    preg_text += f" (Category {item['pregnancy_category_raw']})"
                safety_info.append(f"Pregnancy: {preg_text}")
            
            if item.get('alcohol'):
                alcohol_map = {'X': 'Avoid alcohol', 'N': 'No interaction'}
                safety_info.append(f"Alcohol Interaction: {alcohol_map.get(item['alcohol'], item['alcohol'])}")
            
            if item.get('csa'):
                csa_map = {'N': 'Not a controlled substance'}
                safety_info.append(f"Controlled Substance: {csa_map.get(item['csa'], item['csa'])}")
            
            sections.append('\n'.join(safety_info))
        
        # 5. related drugs
        if item.get('related_drugs'):
            related_section = []
            related_section.append("=== RELATED DRUGS ===")
            related_names = [d['name'] for d in item['related_drugs']]
            related_section.append(', '.join(related_names))
            sections.append('\n'.join(related_section))
        
        # 6. condition summary
        if item.get('condition_summary'):
            condition_section = []
            condition_section.append("=== CONDITION INFORMATION ===")
            condition_section.append(item['condition_summary'])
            sections.append('\n'.join(condition_section))
        
        return '\n\n'.join(sections)


if __name__ == '__main__':
    loader = DataLoader(
        pubmed_path='data/BioASQ/corpus_subset.json',
        openfda_path='data/OpenFDA Drug data/OpenFDA_corpus.json',
        kaggle_path='data/kaggle_drug_data/processed/extracted_docs.json'
    )
    documents = loader.load_all()
    
    for doc in documents[:3]:
        print(f"\n{'='*60}")
        print(f"ID: {doc.doc_id}")
        print(f"Source: {doc.source}")
        print(f"Title: {doc.title}")
        print(f"Content: {doc.content[:200]}...")