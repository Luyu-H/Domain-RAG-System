"""
Answer generation module for RAG system.
Integrates with LLMs to generate answers based on retrieved context.
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import json


class BaseGenerator(ABC):
    """Base class for answer generators"""
    
    @abstractmethod
    def generate(self, query: str, context: List[Dict], **kwargs) -> Dict:
        """
        Generate answer from query and context
        
        Args:
            query: user query
            context: list of retrieved document chunks
            **kwargs: additional parameters
            
        Returns:
            {
                'answer': generated answer text,
                'sources': list of source chunk IDs,
                'metadata': additional metadata
            }
        """
        pass


class AnswerGenerator(BaseGenerator):
    """
    Answer generator using LLM APIs or local models.
    Supports multiple backends: OpenAI, Anthropic, HuggingFace, etc.
    """
    
    def __init__(self, 
                 model_type: str = 'openai',
                 model_name: str = 'gpt-3.5-turbo',
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """
        Args:
            model_type: 'openai', 'anthropic', 'huggingface', 'local'
            model_name: model identifier
            api_key: API key for cloud services
            temperature: sampling temperature
            max_tokens: maximum tokens in response
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on model_type"""
        if self.model_type == 'openai':
            try:
                from openai import OpenAI
                if self.api_key:
                    self._client = OpenAI(api_key=self.api_key)
                else:
                    # Try to use environment variable
                    self._client = OpenAI()
                print(f"Initialized OpenAI client with model: {self.model_name}")
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
                self._client = None
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self._client = None
        
        elif self.model_type == 'anthropic':
            try:
                import anthropic
                if self.api_key:
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                print(f"Initialized Anthropic client with model: {self.model_name}")
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
                self._client = None
        
        elif self.model_type == 'huggingface':
            try:
                from transformers import pipeline
                self._client = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map="auto"
                )
                print(f"Initialized HuggingFace pipeline with model: {self.model_name}")
            except ImportError:
                print("Warning: transformers package not installed. Install with: pip install transformers")
                self._client = None
        
        elif self.model_type == 'local':
            # For local models, you can use Ollama, vLLM, etc.
            print(f"Local model mode: {self.model_name}")
            self._client = None
    
    def _build_prompt(self, query: str, context: List[Dict]) -> str:
        """
        Build prompt from query and context
        
        Args:
            query: user query
            context: list of retrieved chunks with 'text' field
            
        Returns:
            formatted prompt string
        """
        # Format context
        context_texts = []
        for i, chunk in enumerate(context, 1):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            chunk_type = chunk.get('chunk_type', '')
            context_texts.append(
                f"[Document {i}] (Source: {source}, Type: {chunk_type})\n{text}\n"
            )
        
        context_str = "\n".join(context_texts)
        
        # Build prompt
        prompt = f"""You are a medical information assistant. Answer the user's question based on the provided medical documents.

Context Documents:
{context_str}

Question: {query}

Instructions:
1. Answer the question based on the provided context documents.
2. If the answer is not in the context, say so clearly.
3. Cite specific documents when possible (e.g., "According to Document 1...").
4. Be accurate and concise.
5. If multiple documents provide different information, synthesize them appropriately.

Answer:"""
        
        return prompt
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        if not self._client:
            return "Error: OpenAI client not initialized"
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical information assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic API"""
        if not self._client:
            return "Error: Anthropic client not initialized"
        
        try:
            message = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_huggingface(self, prompt: str) -> str:
        """Generate using HuggingFace transformers"""
        if not self._client:
            return "Error: HuggingFace client not initialized"
        
        try:
            outputs = self._client(
                prompt,
                max_length=len(prompt.split()) + self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_full_text=False
            )
            return outputs[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate(self, query: str, context: List[Dict], **kwargs) -> Dict:
        """
        Generate answer from query and context
        
        Args:
            query: user query
            context: list of retrieved document chunks
            **kwargs: additional parameters
            
        Returns:
            {
                'answer': generated answer text,
                'sources': list of source chunk IDs,
                'metadata': additional metadata
            }
        """
        if not context:
            return {
                'answer': "I couldn't find any relevant documents to answer your question.",
                'sources': [],
                'metadata': {}
            }
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate based on model type
        if self.model_type == 'openai':
            answer = self._generate_openai(prompt)
        elif self.model_type == 'anthropic':
            answer = self._generate_anthropic(prompt)
        elif self.model_type == 'huggingface':
            answer = self._generate_huggingface(prompt)
        elif self.model_type == 'local':
            # For local models, implement your own generation logic
            answer = f"[Local model generation not implemented. Model: {self.model_name}]"
        else:
            answer = f"Unsupported model type: {self.model_type}"
        
        # Extract source IDs
        sources = [chunk.get('chunk_id', '') for chunk in context]
        
        return {
            'answer': answer,
            'sources': sources,
            'metadata': {
                'model_type': self.model_type,
                'model_name': self.model_name,
                'num_context_chunks': len(context)
            }
        }


# Simple template-based generator (fallback when LLM not available)
class TemplateGenerator(BaseGenerator):
    """
    Simple template-based answer generator.
    Useful as a fallback when LLM APIs are not available.
    """
    
    def generate(self, query: str, context: List[Dict], **kwargs) -> Dict:
        """Generate answer using simple template"""
        if not context:
            return {
                'answer': "I couldn't find any relevant documents.",
                'sources': [],
                'metadata': {}
            }
        
        # Simple template: concatenate top chunks
        answer_parts = []
        for i, chunk in enumerate(context[:3], 1):  # Use top 3 chunks
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            answer_parts.append(f"[{i}] ({source}): {text[:200]}...")
        
        answer = "\n\n".join(answer_parts)
        sources = [chunk.get('chunk_id', '') for chunk in context]
        
        return {
            'answer': answer,
            'sources': sources,
            'metadata': {
                'generator': 'template',
                'num_chunks': len(context)
            }
        }


if __name__ == '__main__':
    # Test generator
    print("Testing Answer Generator")
    print("=" * 60)
    
    # Mock context
    context = [
        {
            'chunk_id': 'chunk_1',
            'text': 'Aspirin is a medication used to reduce pain, fever, or inflammation.',
            'source': 'pubmed',
            'chunk_type': 'abstract'
        },
        {
            'chunk_id': 'chunk_2',
            'text': 'Common side effects of aspirin include stomach ulcers and bleeding.',
            'source': 'openfda',
            'chunk_type': 'warnings'
        }
    ]
    
    query = "What are the side effects of aspirin?"
    
    # Test template generator
    print("\n[1] Template Generator:")
    template_gen = TemplateGenerator()
    result = template_gen.generate(query, context)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    
    # Test LLM generator (if API key provided)
    print("\n[2] LLM Generator (OpenAI):")
    # Uncomment and add API key to test:
    # llm_gen = AnswerGenerator(model_type='openai', model_name='gpt-3.5-turbo', api_key='YOUR_API_KEY')
    # result = llm_gen.generate(query, context)
    # print(f"Answer: {result['answer']}")

