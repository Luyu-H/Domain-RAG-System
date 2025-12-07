import re
from typing import List, Dict
from dataclasses import dataclass

import scispacy
import spacy

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    chunk_type: str  # 'title', 'abstract', 'basic_info', 'side_effects', etc.
    text: str
    metadata: Dict
    
    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'source': self.source,
            'chunk_type': self.chunk_type,
            'text': self.text,
            'metadata': self.metadata
        }


class DocumentChunker:
    """Chunker for different document sources (PubMed, OpenFDA, Kaggle)"""

    def __init__(self, 
                 max_chunk_size: int = 512,
                 overlap: int = 50,
                 sentence_split: bool = True):
        """
        Args:
            max_chunk_size: maximum characters per chunk, if exceeded, split further
            overlap: number of overlapping characters between chunks
            sentence_split: whether to split by sentences when chunking long sections
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.sentence_split = sentence_split
    
    def chunk_documents(self, documents: List) -> List[Chunk]:
        """Chunk all documents based on their source"""
        all_chunks = []
        
        for doc in documents:
            if doc.source == 'pubmed':
                chunks = self._chunk_pubmed(doc)
            elif doc.source == 'openfda':
                chunks = self._chunk_openfda(doc)
            elif doc.source == 'kaggle':
                chunks = self._chunk_kaggle(doc)
            else:
                chunks = self._chunk_generic(doc)
            
            all_chunks.extend(chunks)
        
        print(f"Generated {len(all_chunks)} chunks in total")
        return all_chunks

    def _chunk_pubmed(self, doc) -> List[Chunk]:
        """
        Chunking strategy for PubMed documents:
            1. Title as one chunk
            2. Abstract as one or more chunks (split by sentences if too long)
        """
        chunks = []
        
        # Chunk 1: Title
        if doc.title:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_title",
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_type='title',
                text=doc.title,
                metadata={**doc.metadata, 'section': 'title'}
            ))
        
        # Chunk 2-N: Abstract
        if doc.content:
            if len(doc.content) <= self.max_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_abstract",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunk_type='abstract',
                    text=doc.content,
                    metadata={**doc.metadata, 'section': 'abstract'}
                ))
            else:
                abstract_chunks = self._split_by_sentences(
                    doc.content,
                    doc.doc_id,
                    doc.source,
                    'abstract',
                    doc.metadata
                )
                chunks.extend(abstract_chunks)
        
        return chunks

    def _chunk_openfda(self, doc) -> List[Chunk]:
        """
        Chunking strategy for OpenFDA documents:
            Chunks based on sections:
            - Drug names
            - Active ingredients  
            - Indications and usage
            - Dosage and administration
            - Warning information
            - Purpose
            - Pregnancy information
            - Overdosage
            - Inactive ingredients
            - Other safety information
        """
        chunks = []
        
        # title: brand_name + generic_name
        if doc.title:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_title",
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_type='title',
                text=doc.title,
                metadata={**doc.metadata, 'section': 'title'}
            ))
        
        # parse sections from content
        if doc.content:
            section_chunks = self._parse_openfda_sections(doc)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _parse_openfda_sections(self, doc) -> List[Chunk]:
        """Parse OpenFDA sections from content"""
        chunks = []
      
        section_patterns = [
            ('drug_names', r'Drug names?:'),
            ('active_ingredients', r'Active ingredients?:?'),
            ('indications', r'Indications and usage:?'),
            ('dosage', r'Dosage and administration:?'),
            ('warnings', r'Warning information:?'),
            ('purpose', r'Purpose:?'),
            ('pregnancy', r'Pregnancy information:?'),
            ('overdosage', r'Overdosage:?'),
            ('inactive_ingredients', r'Inactive ingredients?:?'),
            ('other_info', r'Other (?:safety )?information:?')
        ]
        
        text = doc.content
        
        section_positions = []
        for section_type, pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_positions.append((match.start(), section_type, match.group()))
        
        section_positions.sort(key=lambda x: x[0])
        
        # extract sections
        for i, (start, section_type, section_header) in enumerate(section_positions):
            if i < len(section_positions) - 1:
                end = section_positions[i + 1][0]
            else:
                end = len(text)
            
            section_text = text[start:end].strip()
            section_content = section_text[len(section_header):].strip()
            if not section_content:
                continue
            
            # if section is too long, split further
            if len(section_content) > self.max_chunk_size:
                sub_chunks = self._split_long_section(
                    section_content,
                    doc.doc_id,
                    doc.source,
                    section_type,
                    doc.metadata
                )
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_{section_type}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunk_type=section_type,
                    text=section_content,
                    metadata={**doc.metadata, 'section': section_type}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_kaggle(self, doc) -> List[Chunk]:
        """
        Kaggle document chunking strategy:
            similar to OpenFDA, based on sections marked by === SECTION NAME ===
        """
        chunks = []
        
        # Title chunk
        if doc.title:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_title",
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_type='title',
                text=doc.title,
                metadata={**doc.metadata, 'section': 'title'}
            ))
        
        # parse sections in content
        if doc.content:
            section_chunks = self._parse_kaggle_sections(doc)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _parse_kaggle_sections(self, doc) -> List[Chunk]:
        """Parse Kaggle sections from content"""
        chunks = []
        
        # split by section headers
        sections = re.split(r'(===\s+[A-Z\s()]+\s+===)', doc.content)
        
        current_section_type = None
        current_section_content = []
        
        for part in sections:
            part = part.strip()
            if not part:
                continue
            
            # check if part is a section header
            section_match = re.match(r'===\s+([A-Z\s()]+)\s+===', part)
            
            if section_match:
                # save previous section
                if current_section_type and current_section_content:
                    self._save_kaggle_section(
                        chunks,
                        doc,
                        current_section_type,
                        '\n'.join(current_section_content)
                    )
                
                # start new section
                section_name = section_match.group(1).strip()
                current_section_type = self._normalize_section_name(section_name)
                current_section_content = []
            else:
                current_section_content.append(part)
        
        # save last section
        if current_section_type and current_section_content:
            self._save_kaggle_section(
                chunks,
                doc,
                current_section_type,
                '\n'.join(current_section_content)
            )
        
        return chunks
    
    def _normalize_section_name(self, section_name: str) -> str:
        """Standardize section names to chunk types"""
        name_map = {
            'BASIC INFORMATION': 'basic_info',
            'BRAND NAMES': 'brand_names',
            'SIDE EFFECTS': 'side_effects',
            'SIDE EFFECTS (RAW)': 'side_effects_raw',
            'SAFETY INFORMATION': 'safety_info',
            'RELATED DRUGS': 'related_drugs',
            'CONDITION INFORMATION': 'condition_info'
        }
        return name_map.get(section_name, section_name.lower().replace(' ', '_'))
    
    def _save_kaggle_section(self, chunks: List, doc, section_type: str, content: str):
        """save a Kaggle section as one or more chunks"""
        content = content.strip()
        if not content:
            return
    
        if len(content) > self.max_chunk_size:
            sub_chunks = self._split_long_section(
                content,
                doc.doc_id,
                doc.source,
                section_type,
                doc.metadata
            )
            chunks.extend(sub_chunks)
        else:
            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_{section_type}",
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_type=section_type,
                text=content,
                metadata={**doc.metadata, 'section': section_type}
            )
            chunks.append(chunk)
    
    
    # help functions for splitting long sections
    def _split_long_section(self, text: str, doc_id: str, source: str, 
                           section_type: str, metadata: Dict) -> List[Chunk]:
        """Split a long section into smaller chunks"""
        if self.sentence_split:
            return self._split_by_sentences(text, doc_id, source, section_type, metadata)
        else:
            return self._split_by_words(text, doc_id, source, section_type, metadata)
    
    def _split_by_sentences(self, text: str, doc_id: str, source: str,
                           section_type: str, metadata: Dict) -> List[Chunk]:
        """Split by sentences (with overlap)"""

        # use SciSpacy for sentence segmentation instead of regex simple split
        nlp = spacy.load("en_core_sci_sm")   # Sci / Bio nlp model
        doc = nlp(text)

        sentences = [sent.text for sent in doc.sents]

        # sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # if adding this sentence exceeds max chunk size, save current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{section_type}_{len(chunks)}",
                    doc_id=doc_id,
                    source=source,
                    chunk_type=section_type,
                    text=chunk_text,
                    metadata={**metadata, 'section': section_type, 'chunk_index': len(chunks)}
                ))
                
                # start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > 1:
                    # save last sentence as overlap
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # save any remaining sentences as last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{section_type}_{len(chunks)}",
                doc_id=doc_id,
                source=source,
                chunk_type=section_type,
                text=chunk_text,
                metadata={**metadata, 'section': section_type, 'chunk_index': len(chunks)}
            ))
        
        return chunks
    
    def _split_by_words(self, text: str, doc_id: str, source: str,
                       section_type: str, metadata: Dict) -> List[Chunk]:
        """Chunk by words (simple split)"""
        words = text.split()
        chunks = []
        
        # estimate average word length as 5 characters
        words_per_chunk = self.max_chunk_size // 5
        overlap_words = self.overlap // 5
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{section_type}_{len(chunks)}",
                doc_id=doc_id,
                source=source,
                chunk_type=section_type,
                text=chunk_text,
                metadata={**metadata, 'section': section_type, 'chunk_index': len(chunks)}
            ))
        
        return chunks
    
    def _chunk_generic(self, doc) -> List[Chunk]:
        """Generic chunking strategy: combine title and content, split by words"""
        full_text = f"{doc.title}\n{doc.content}"
        return self._split_by_words(full_text, doc.doc_id, doc.source, 'generic', doc.metadata)


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    
    from data_loader import DataLoader

    print("Loading data...")
    loader = DataLoader(
        pubmed_path='data/BioASQ/corpus_subset.json',
        openfda_path='data/OpenFDA Drug data/OpenFDA_corpus.json',
        kaggle_path='data/kaggle_drug_data/processed/extracted_docs.json'
    )
    documents = loader.load_all()
    
    print("\nChunking...")
    chunker = DocumentChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    print("\n" + "="*60)
    print("Chunk statistics:")
    print("="*60)
    
    source_stats = {}
    for chunk in chunks:
        source_stats[chunk.source] = source_stats.get(chunk.source, 0) + 1
    
    print("\nSource:")
    for source, count in sorted(source_stats.items()):
        print(f"  {source}: {count} chunks")

    type_stats = {}
    for chunk in chunks:
        type_stats[chunk.chunk_type] = type_stats.get(chunk.chunk_type, 0) + 1
    
    print("\Chunk type:")
    for ctype, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ctype}: {count} chunks")
    
    print("\n" + "="*60)
    print("Sample chunks (one for each source):")
    print("="*60)
    
    shown_sources = set()
    for chunk in chunks:
        if chunk.source not in shown_sources:
            print(f"\n【{chunk.source.upper()}】")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Type: {chunk.chunk_type}")
            print(f"Text: {chunk.text[:200]}...")
            print(f"Metadata: {list(chunk.metadata.keys())}")
            shown_sources.add(chunk.source)
            
            if len(shown_sources) >= 3:
                break

    print("\n" + "="*60)
    print("Sample chunks for each chunk_type:")
    print("="*60)
    
    shown_types = set()
    for chunk in chunks:
        if chunk.chunk_type not in shown_types:
            print(f"\n【{chunk.chunk_type}】")
            print(f"Source: {chunk.source}")
            print(f"Text: {chunk.text[:150]}...")
            shown_types.add(chunk.chunk_type)
            
            if len(shown_types) >= 10:
                break