"""
Simple knowledge base query tool that retrieves documents from S3 vectors.
"""
import boto3
import json
from typing import List, Dict, Any
from strands import tool


class KnowledgeBaseQuery:
    """Simple knowledge base query system that retrieves documents from S3 vectors."""
    
    def __init__(self, region_name: str = "eu-central-1"):
        """Initialize the query system with AWS clients.
        
        Args:
            region_name: AWS region for the services
        """
        self.region_name = region_name
        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        self.s3vectors = boto3.client("s3vectors", region_name=region_name)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the input text using Titan Text Embeddings V2.
        
        Args:
            text: Input text to embed
            
        Returns:
            The embedding vector as a list of floats
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": text})
            )
            
            model_response = json.loads(response["body"].read())
            return model_response["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def _query_vectors(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the S3 vectors index with the embedding.
        
        Args:
            embedding: The query embedding vector
            top_k: Maximum number of results to return
            
        Returns:
            List of retrieved documents with metadata and distances
            
        Raises:
            Exception: If vector query fails
        """
        try:
            query_params = {
                "vectorBucketName": "acrelec-model-eval",
                "indexName": "acrelec-model-eval-index",
                "queryVector": {"float32": embedding},
                "topK": top_k,
                "returnDistance": True,
                "returnMetadata": True,
            }
            
            response = self.s3vectors.query_vectors(**query_params)
            return response.get("vectors", [])
        except Exception as e:
            raise Exception(f"Failed to query vectors: {str(e)}")
    
    @tool
    def ask_knowledgebase(self, query: str) -> str:
        """Query the knowledge base with the provided question."""
        try:
            # Step 1: Generate embedding for the query
            embedding = self._generate_embedding(query)
            
            # Step 2: Query the vector database
            vectors = self._query_vectors(embedding, top_k=5)
            
            # Step 3: Format and return the results
            if not vectors:
                return f"No relevant documents found in the knowledge base for query: '{query}'"
            
            results = []
            results.append(f"Found {len(vectors)} relevant documents for query: '{query}'\n")
            
            structured_docs = []
            for i, vector in enumerate(vectors, 1):
                metadata = vector.get("metadata", {})
                distance = vector.get("distance", None)
                file_id = vector.get("key", None)
                
                if isinstance(distance, (int, float)):
                    relevance_score = 1 - distance
                    relevance_str = f"{relevance_score:.3f}"
                else:
                    relevance_str = "unknown"

                results.append(f"--- Document {i} (Relevance Score: {relevance_str}) ---")

                # Add file_id information (strip everything after and including the dash)
                clean_file_id = None
                if file_id:
                    clean_file_id = file_id.split('-')[0] if '-' in file_id else file_id
                    results.append(f"file_id: {clean_file_id}")

                # Add metadata information
                doc_meta_output = {}
                if metadata:
                    for key, value in metadata.items():
                        results.append(f"{key.title()}: {value}")
                        doc_meta_output[key] = value

                structured_docs.append({
                    "file_id": clean_file_id,
                    "relevance_score": relevance_str,
                    "distance": distance,
                    "metadata": doc_meta_output
                })

                results.append("-" * 50)
            
            transparency = {
                "query": query,
                "retrieved_count": len(structured_docs),
                "documents": structured_docs
            }

            return "\n".join(results) + "\n\n```json\n" + json.dumps(transparency, ensure_ascii=False, indent=2) + "\n```"
            
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"


# Create a global instance for easy use
_kb_query = KnowledgeBaseQuery()

# Export the tool function for direct use
ask_knowledgebase = _kb_query.ask_knowledgebase


