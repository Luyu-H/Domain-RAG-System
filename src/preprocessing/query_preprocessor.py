import re
from typing import List, Dict
import scispacy
import spacy
from .medical_term_normalizer import MedicalTermNormalizer


class QueryPreprocessor:
    """
    Query preprocessing pipeline:
        1. Clean query
        2. Extract medical entities
        3. Normalize medical terms
    """
    
    def __init__(self, 
                 medical_normalizer: MedicalTermNormalizer = None,
                 use_scispacy: bool = True):
        """
        Args:
            medical_normalizer: MedicalTermNormalizer instance
            use_scispacy: whether to use scispaCy for NER
        """
        self.normalizer = medical_normalizer or MedicalTermNormalizer()
        
        # Load scispaCy model for medical NER
        self.nlp = None
        if use_scispacy:
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                try:
                    self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                except:
                    print("UMLS linker not available, using basic NER only")
            except:
                print("Warning: scispaCy model not found, using basic preprocessing only")
                self.nlp = None
    
    def preprocess(self, query: str, 
                   extract_entities: bool = True,
                   normalize_terms: bool = True,
                   expand_abbreviations: bool = True) -> Dict:
        """
        Preprocess query
        
        Returns:
            {
                'original': original query,
                'cleaned': cleaned query,
                'normalized': normalized query,
                'entities': extracted entities,
                'medical_terms': extracted medical terms
            }
        """
        result = {
            'original': query,
            'cleaned': '',
            'normalized': '',
            'entities': [],
            'medical_terms': {}
        }
        
        # Clean query
        cleaned = self._clean_query(query)
        result['cleaned'] = cleaned
        
        # Extract entities (using scispaCy)
        if extract_entities and self.nlp:
            entities = self._extract_entities(cleaned)
            result['entities'] = entities
        
        # Normalize medical terms
        if normalize_terms:
            normalized = self.normalizer.normalize_text(
                cleaned, 
                expand_abbreviations=expand_abbreviations
            )
            result['normalized'] = normalized
            
            # Extract medical terms
            medical_terms = self.normalizer.extract_medical_terms(cleaned)
            result['medical_terms'] = medical_terms
        else:
            result['normalized'] = cleaned
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """Basic query cleaning"""
        if not query:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query)
        
        # Remove special characters (but keep medical symbols like +, -, /)
        # Keep: alphanumeric, spaces, and medical-relevant punctuation
        cleaned = re.sub(r'[^\w\s\-\+/().,?]', '', cleaned)
        
        # Trim
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using scispaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Add UMLS linking if available
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                entity_info['umls_id'] = ent._.kb_ents[0][0]  # Top match
            
            entities.append(entity_info)
        
        return entities
    
    def get_query_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        (removing stopwords but keeping medical terms)
        """
        if not self.nlp:
            # Simple split if NLP not available
            return query.lower().split()
        
        doc = self.nlp(query)
        
        keywords = []
        for token in doc:
            # Keep if: 1. Not a stopword, OR 2. Is a medical term (check with normalizer)
            if not token.is_stop or token.text.lower() in self.normalizer.medical_stopwords:
                if token.is_alpha:  # Only alphabetic tokens
                    keywords.append(token.lemma_.lower())
        
        return keywords


if __name__ == '__main__':
    preprocessor = QueryPreprocessor()
    
    test_queries = [
        "What are the side effects of aspirin?",
        "Treatment for pt with MI and HTN",
        "Drugs for diabetes and GERD",
        "How to treat heart attack patients?",
    ]
    
    print("Testing Query Preprocessor")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = preprocessor.preprocess(query)
        
        print(f"Cleaned: {result['cleaned']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Medical terms: {result['medical_terms']}")
        print(f"Entities: {[e['text'] + ' (' + e['label'] + ')' for e in result['entities']]}")
        
        keywords = preprocessor.get_query_keywords(query)
        print(f"Keywords: {keywords}")