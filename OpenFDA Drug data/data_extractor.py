#!/usr/bin/env python3
"""
OpenFDA Drug Label Data Extraction and Cleaning Script
Prepare data for vector storage system testing
"""

import json
import re
from typing import List, Dict, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugDataExtractor:
    """Drug data extractor"""
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.extracted_data = []
        
    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text
    
    def extract_drug_documents(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Extract drug document data"""
        logger.info(f"Starting drug data extraction, limited to {limit} records")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            logger.info(f"Found {len(results)} original records")
            
            processed_count = 0
            for record in results[:limit]:
                try:
                    # Extract key fields
                    doc = self._extract_document(record)
                    if doc:
                        self.extracted_data.append(doc)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logger.info(f"Processed {processed_count} records")
                            
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(self.extracted_data)} drug documents")
            return self.extracted_data
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []
    
    def _extract_document(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document from single record"""
        
        # Extract basic information
        doc_id = record.get('id', '')
        set_id = record.get('set_id', '')
        
        # Extract text content
        text_parts = []
        
        # Drug names
        brand_name = self._extract_field(record, 'openfda', 'brand_name')
        generic_name = self._extract_field(record, 'openfda', 'generic_name')
        
        if brand_name or generic_name:
            text_parts.append(f"Drug names: {brand_name} ({generic_name})")
        
        # Active ingredients
        active_ingredients = self._extract_field(record, 'active_ingredient')
        if active_ingredients:
            text_parts.append(f"Active ingredients: {active_ingredients}")
        
        # Indications and usage
        indications = self._extract_field(record, 'indications_and_usage')
        if indications:
            text_parts.append(f"Indications and usage: {indications}")
        
        # Dosage and administration
        dosage = self._extract_field(record, 'dosage_and_administration')
        if dosage:
            text_parts.append(f"Dosage and administration: {dosage}")
        
        # Warning information
        warnings = self._extract_field(record, 'warnings')
        if warnings:
            text_parts.append(f"Warning information: {warnings}")
        
        # Purpose
        purpose = self._extract_field(record, 'purpose')
        if purpose:
            text_parts.append(f"Purpose: {purpose}")
        
        # Pregnancy information
        pregnancy_info = self._extract_field(record, 'pregnancy_or_breast_feeding')
        if pregnancy_info:
            text_parts.append(f"Pregnancy information: {pregnancy_info}")
        
        # Overdosage information
        overdosage = self._extract_field(record, 'overdosage')
        if overdosage:
            text_parts.append(f"Overdosage: {overdosage}")
        
        # Inactive ingredients
        inactive_ingredients = self._extract_field(record, 'inactive_ingredient')
        if inactive_ingredients:
            text_parts.append(f"Inactive ingredients: {inactive_ingredients}")
        
        # Other safety information
        safety_info = self._extract_field(record, 'other_safety_information')
        if safety_info:
            text_parts.append(f"Other safety information: {safety_info}")
        
        if not text_parts:
            return None
        
        # Combine complete document
        full_text = "\n".join(text_parts)
        
        # Create document object
        document = {
            'id': doc_id,
            'set_id': set_id,
            'brand_name': brand_name,
            'generic_name': generic_name,
            'text': full_text,
            'metadata': {
                'effective_time': record.get('effective_time', ''),
                'version': record.get('version', ''),
                'has_active_ingredients': bool(active_ingredients),
                'has_warnings': bool(warnings),
                'has_dosage': bool(dosage),
                'has_indications': bool(indications),
                'text_length': len(full_text)
            }
        }
        
        return document
    
    def _extract_field(self, record: Dict[str, Any], *keys) -> str:
        """Extract nested field data"""
        current = record
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return ""
        
        if isinstance(current, list):
            # Merge text from list
            text = ' '.join([str(item) for item in current if item])
        else:
            text = str(current) if current else ""
        
        return self.clean_text(text)
    
    def save_extracted_data(self, output_file: str):
        """Save extracted data"""
        if not self.extracted_data:
            logger.warning("No data to save")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Extracted data saved to: {output_file}")
        
        # Save plain text file for vectorization
        text_file = output_file.replace('.json', '_texts.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            for doc in self.extracted_data:
                f.write(f"ID: {doc['id']}\n")
                f.write(f"Brand: {doc['brand_name']}\n")
                f.write(f"Generic: {doc['generic_name']}\n")
                f.write(f"Text: {doc['text']}\n")
                f.write("-" * 80 + "\n")
        
        logger.info(f"Plain text data saved to: {text_file}")

def main():
    """Main function"""
    input_file = "drug-label-0001-of-0013.json"
    output_file = "extracted_drug_documents.json"
    
    # Extract data
    extractor = DrugDataExtractor(input_file)
    extractor.extract_drug_documents(limit=1000)  # Limit to 1000 records for testing
    extractor.save_extracted_data(output_file)
    
    # Display statistics
    print(f"\nData extraction completed:")
    print(f"- Total documents: {len(extractor.extracted_data)}")
    if extractor.extracted_data:
        avg_length = sum(doc['metadata']['text_length'] for doc in extractor.extracted_data) / len(extractor.extracted_data)
        print(f"- Average document length: {avg_length:.0f} characters")
        
        # Count documents with various information
        stats = {
            'Has Active ingredients': sum(1 for doc in extractor.extracted_data if doc['metadata']['has_active_ingredients']),
            'Has Warning information': sum(1 for doc in extractor.extracted_data if doc['metadata']['has_warnings']),
            'Has Dosage Info': sum(1 for doc in extractor.extracted_data if doc['metadata']['has_dosage']),
            'Has Indications': sum(1 for doc in extractor.extracted_data if doc['metadata']['has_indications'])
        }
        
        print(f"- Document feature statistics:")
        for feature, count in stats.items():
            print(f"  {feature}: {count} ({count/len(extractor.extracted_data)*100:.1f}%)")

if __name__ == "__main__":
    main()
