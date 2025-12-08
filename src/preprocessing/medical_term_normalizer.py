import re
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path
from collections import defaultdict


class MedicalTermNormalizer:
    """
    Medical terminology normalization using:
        1. Drug abbreviations and synonyms
        2. Medical condition synonyms
        3. Generic name <-> Brand name mapping
    """
    
    def __init__(self, abbreviations_path: str = None, drug_mapping_path: str = None):
        """
        Args:
            abbreviations_path: path to medical abbreviations dictionary
            drug_mapping_path: path to drug name mapping (generic <-> brand)
        """
        self.abbreviations = self._load_abbreviations(abbreviations_path)

        # Drug name mapping (will be built from data)
        self.drug_mapping = self._load_drug_mapping(drug_mapping_path)
        self.condition_synonyms = self._build_condition_synonyms()
        
        # Common medical terms that should be preserved
        self.medical_stopwords = self._build_medical_stopwords()
        
    def normalize_text(self, text: str, expand_abbreviations: bool = True) -> str:
        """
        Normalize medical text
        
        Args:
            text: input text
            expand_abbreviations: whether to expand abbreviations
            
        Returns:
            normalized text
        """
        if not text:
            return text
        
        # Convert to lowercase for matching (but preserve original for output)
        normalized = text
        
        # Expand abbreviations
        if expand_abbreviations:
            normalized = self._expand_abbreviations(normalized)
        
        # Normalize drug names (brand -> generic)
        normalized = self._normalize_drug_names(normalized)
        # Normalize medical conditions
        normalized = self._normalize_conditions(normalized)
        
        return normalized
    
    def extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical terms from text
        
        Returns:
            {
                'drugs': [...],
                'conditions': [...],
                'abbreviations': [...]
            }
        """
        result = {
            'drugs': [],
            'conditions': [],
            'abbreviations': []
        }
        
        text_lower = text.lower()
        
        # Extract drugs
        for generic, brands in self.drug_mapping.items():
            if generic in text_lower:
                result['drugs'].append(generic)
            for brand in brands:
                if brand.lower() in text_lower:
                    result['drugs'].append(f"{brand} ({generic})")
        
        # Extract abbreviations
        for abbr in self.abbreviations.keys():
            # Use word boundary to match abbreviations
            if re.search(r'\b' + re.escape(abbr) + r'\b', text, re.IGNORECASE):
                result['abbreviations'].append(abbr)
        
        return result
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get all synonyms for a medical term"""
        term_lower = term.lower()
        synonyms = set()
        
        # Check drug names
        if term_lower in self.drug_mapping:
            synonyms.update(self.drug_mapping[term_lower])
        
        # Check if it's a brand name
        for generic, brands in self.drug_mapping.items():
            if term_lower in [b.lower() for b in brands]:
                synonyms.add(generic)
                synonyms.update(brands)
        
        # Check condition synonyms
        for condition, syns in self.condition_synonyms.items():
            if term_lower in [condition.lower()] + [s.lower() for s in syns]:
                synonyms.add(condition)
                synonyms.update(syns)
        
        # Check abbreviations
        if term_lower in self.abbreviations:
            synonyms.add(self.abbreviations[term_lower])
        
        # Remove the original term and return
        synonyms.discard(term)
        synonyms.discard(term_lower)
        
        return list(synonyms)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text"""
        # Sort by length (longer first to avoid partial matches)
        sorted_abbr = sorted(self.abbreviations.items(), 
                            key=lambda x: len(x[0]), 
                            reverse=True)
        
        for abbr, expansion in sorted_abbr:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_drug_names(self, text: str) -> str:
        """Normalize drug names (brand -> generic)"""
        text_lower = text.lower()
        
        for generic, brands in self.drug_mapping.items():
            for brand in brands:
                if brand.lower() in text_lower:
                    # Replace brand name with "generic_name (brand_name)"
                    pattern = r'\b' + re.escape(brand) + r'\b'
                    replacement = f"{generic} ({brand})"
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_conditions(self, text: str) -> str:
        """Normalize medical condition names"""
        for standard, synonyms in self.condition_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in text.lower():
                    pattern = r'\b' + re.escape(synonym) + r'\b'
                    text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        
        return text
    
    def _load_abbreviations(self, filepath: str = None) -> Dict[str, str]:
        """Load medical abbreviations dictionary"""
        if filepath and Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Default common medical abbreviations
        return {
            # Cardiovascular
            'MI': 'myocardial infarction',
            'CHF': 'congestive heart failure',
            'HTN': 'hypertension',
            'CAD': 'coronary artery disease',
            'AF': 'atrial fibrillation',
            'CVA': 'cerebrovascular accident',
            
            # Diabetes
            'DM': 'diabetes mellitus',
            'T1DM': 'type 1 diabetes mellitus',
            'T2DM': 'type 2 diabetes mellitus',
            'HbA1c': 'hemoglobin A1c',
            
            # Respiratory
            'COPD': 'chronic obstructive pulmonary disease',
            'URI': 'upper respiratory infection',
            'SOB': 'shortness of breath',
            
            # Gastrointestinal
            'GERD': 'gastroesophageal reflux disease',
            'IBD': 'inflammatory bowel disease',
            'IBS': 'irritable bowel syndrome',
            
            # General
            'Rx': 'prescription',
            'Sx': 'symptoms',
            'Tx': 'treatment',
            'Dx': 'diagnosis',
            'Hx': 'history',
            'pt': 'patient',
            'w/': 'with',
            'w/o': 'without',
            
            # Lab values
            'WBC': 'white blood cell count',
            'RBC': 'red blood cell count',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'BUN': 'blood urea nitrogen',
            
            # Medications
            'NSAID': 'nonsteroidal anti-inflammatory drug',
            'ACE': 'angiotensin-converting enzyme',
            'ARB': 'angiotensin receptor blocker',
            'SSRI': 'selective serotonin reuptake inhibitor',
            'PPI': 'proton pump inhibitor',
        }
    
    def _load_drug_mapping(self, filepath: str = None) -> Dict[str, List[str]]:
        """
        Load drug name mapping (generic -> brand names)
        If file doesn't exist, return empty dict (will be built from data)
        """
        if filepath and Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Will be populated from the actual data
        return {}
    
    def build_drug_mapping_from_data(self, documents: List) -> None:
        """
        Build drug name mapping from loaded documents
        
        Args:
            documents: List of Document objects
        """
        drug_map = defaultdict(set)
        
        for doc in documents:
            # From OpenFDA
            if doc.source == 'openfda':
                generic = doc.metadata.get('generic_name', '').lower().strip()
                brand = doc.metadata.get('brand_name', '').strip()
                
                if generic and brand:
                    drug_map[generic].add(brand)
            
            # From Kaggle
            elif doc.source == 'kaggle':
                generic = doc.metadata.get('generic_name', '').lower().strip()
                drug_name = doc.metadata.get('drug_name', '').strip()
                brand_names = doc.metadata.get('brand_names', [])
                
                if generic and drug_name and generic != drug_name.lower():
                    drug_map[generic].add(drug_name)
                
                if generic and brand_names:
                    for brand in brand_names:
                        if brand.lower() != generic:
                            drug_map[generic].add(brand)
        
        # Convert sets to lists
        self.drug_mapping = {k: list(v) for k, v in drug_map.items() if v}
        
        print(f"Built drug mapping: {len(self.drug_mapping)} generic drugs")
    
    def save_drug_mapping(self, filepath: str):
        """Save drug mapping to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.drug_mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved drug mapping to {filepath}")
    
    def _build_condition_synonyms(self) -> Dict[str, List[str]]:
        """Build medical condition synonyms dictionary"""
        return {
            'diabetes mellitus': ['diabetes', 'DM', 'sugar disease'],
            'hypertension': ['high blood pressure', 'HTN', 'elevated blood pressure'],
            'myocardial infarction': ['heart attack', 'MI', 'cardiac arrest'],
            'cerebrovascular accident': ['stroke', 'CVA', 'brain attack'],
            'gastroesophageal reflux disease': ['GERD', 'acid reflux', 'heartburn'],
            'chronic obstructive pulmonary disease': ['COPD', 'emphysema', 'chronic bronchitis'],
            'influenza': ['flu', 'grippe'],
            'pneumonia': ['lung infection', 'pulmonary infection'],
            'depression': ['major depressive disorder', 'MDD', 'clinical depression'],
            'anxiety': ['anxiety disorder', 'GAD', 'generalized anxiety disorder'],
        }
    
    def _build_medical_stopwords(self) -> Set[str]:
        """Build set of medical terms that should NOT be removed as stopwords"""
        return {
            'patient', 'treatment', 'drug', 'medication', 'dose', 'side effect',
            'symptom', 'disease', 'condition', 'diagnosis', 'prescription',
            'therapy', 'adverse', 'reaction', 'contraindication', 'warning'
        }


if __name__ == '__main__':
    normalizer = MedicalTermNormalizer()
    
    test_texts = [
        "Patient has MI and HTN",
        "Prescribed Tylenol for pain",
        "Diabetes pt w/ GERD symptoms",
        "Treatment for heart attack patients"
    ]
    
    print("Testing Medical Term Normalizer")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        normalized = normalizer.normalize_text(text)
        print(f"Normalized: {normalized}")
        
        terms = normalizer.extract_medical_terms(text)
        print(f"Extracted terms: {terms}")