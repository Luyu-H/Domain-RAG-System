#!/usr/bin/env python3
"""
Create realistic and non-trivial test queries
For testing retrieval effectiveness of different vector storage systems
"""

import json
from typing import List, Dict, Any

class TestQueryGenerator:
    """Test query generator"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.documents = []
        self.load_data()
    
    def load_data(self):
        """Load extracted data"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents")
        except Exception as e:
            print(f"Failed to load data: {e}")
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create 4 realistic and non-trivial test queries"""
        
        queries = [
            {
                "id": "query_1",
                "query": "What medications are used to treat high blood pressure and cardiovascular conditions?",
                "description": "Cardiovascular disease treatment drug query",
                "expected_fields": ["indications", "purpose", "active_ingredients"],
                "difficulty": "medium",
                "type": "therapeutic_use"
            },
            {
                "id": "query_2", 
                "query": "Find drugs that contain acetaminophen and are safe for pregnant women",
                "description": "Pregnancy-safe acetaminophen drug query",
                "expected_fields": ["active_ingredients", "pregnancy_or_breast_feeding", "warnings"],
                "difficulty": "high",
                "type": "safety_ingredient"
            },
            {
                "id": "query_3",
                "query": "What are the side effects and warnings for medications used to treat depression and anxiety?",
                "description": "Antidepressant anxiety drug side effects query",
                "expected_fields": ["warnings", "indications", "overdosage"],
                "difficulty": "high", 
                "type": "safety_warnings"
            },
            {
                "id": "query_4",
                "query": "Find over-the-counter pain relievers that can be taken with food and have minimal drug interactions",
                "description": "OTC pain reliever safety and interaction query",
                "expected_fields": ["purpose", "dosage_and_administration", "warnings", "other_safety_information"],
                "difficulty": "very_high",
                "type": "complex_safety"
            }
        ]
        
        return queries
    
    def create_ground_truth(self, queries: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Create ground truth for each query (based on keyword matching)"""
        ground_truth = {}
        
        for query in queries:
            relevant_docs = []
            query_text = query["query"].lower()
            
            for doc in self.documents:
                doc_text = doc["text"].lower()
                score = 0
                
                # Calculate relevance score based on query type
                if query["type"] == "therapeutic_use":
                    # Find cardiovascular-related keywords
                    cardio_keywords = ["blood pressure", "cardiovascular", "heart", "hypertension", "cardiac"]
                    for keyword in cardio_keywords:
                        if keyword in doc_text:
                            score += 1
                
                elif query["type"] == "safety_ingredient":
                    # Find acetaminophen and pregnancy safety information
                    if "acetaminophen" in doc_text or "paracetamol" in doc_text:
                        score += 2
                    if "pregnant" in doc_text or "pregnancy" in doc_text:
                        score += 1
                    if "safe" in doc_text:
                        score += 1
                
                elif query["type"] == "safety_warnings":
                    # Find antidepressant and side effect information
                    depression_keywords = ["depression", "anxiety", "antidepressant", "mental health"]
                    warning_keywords = ["side effect", "warning", "adverse", "contraindication"]
                    
                    for keyword in depression_keywords:
                        if keyword in doc_text:
                            score += 1
                    for keyword in warning_keywords:
                        if keyword in doc_text:
                            score += 1
                
                elif query["type"] == "complex_safety":
                    # Find pain relievers, food interactions, drug interactions
                    pain_keywords = ["pain", "analgesic", "relief", "ache"]
                    food_keywords = ["food", "meal", "empty stomach", "with food"]
                    interaction_keywords = ["interaction", "contraindication", "avoid"]
                    
                    for keyword in pain_keywords:
                        if keyword in doc_text:
                            score += 1
                    for keyword in food_keywords:
                        if keyword in doc_text:
                            score += 1
                    for keyword in interaction_keywords:
                        if keyword in doc_text:
                            score += 1
                
                # If score > 0, consider as relevant document
                if score > 0:
                    relevant_docs.append({
                        "doc_id": doc["id"],
                        "score": score,
                        "brand_name": doc["brand_name"],
                        "generic_name": doc["generic_name"]
                    })
            
            # Sort by score, take top 10 as ground truth
            relevant_docs.sort(key=lambda x: x["score"], reverse=True)
            ground_truth[query["id"]] = [doc["doc_id"] for doc in relevant_docs[:10]]
        
        return ground_truth
    
    def save_test_queries(self, output_file: str):
        """Save test queries and ground truth"""
        queries = self.create_test_queries()
        ground_truth = self.create_ground_truth(queries)
        
        test_data = {
            "queries": queries,
            "ground_truth": ground_truth,
            "total_documents": len(self.documents),
            "description": "OpenFDA Drug Label Data Test Query Set"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Test queries saved to: {output_file}")
        
        # Print query statistics
        print(f"\nTest query statistics:")
        for query in queries:
            gt_count = len(ground_truth[query["id"]])
            print(f"- {query['id']}: {query['description']} (Difficulty: {query['difficulty']}, Ground Truth: {gt_count} documents)")

def main():
    """Main function"""
    data_file = "extracted_drug_documents.json"
    output_file = "test_queries.json"
    
    generator = TestQueryGenerator(data_file)
    generator.save_test_queries(output_file)

if __name__ == "__main__":
    main()
