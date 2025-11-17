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
        # Store the last retrieved documents for server access
        self.last_documents = []
    
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
            print(f"DEBUG: Embedding generation failed: {type(e).__name__}: {str(e)}")
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
                "vectorBucketName": "auvaria-vector-bucket",
                "indexName": "auvaria-vector-index",
                "queryVector": {"float32": embedding},
                "topK": top_k,
                "returnDistance": True,
                "returnMetadata": True,
            }
            
            response = self.s3vectors.query_vectors(**query_params)
            print(f"DEBUG: S3 vectors response keys: {list(response.keys())}")
            return response.get("vectors", [])
        except Exception as e:
            print(f"DEBUG: Vector query failed: {type(e).__name__}: {str(e)}")
            raise Exception(f"Failed to query vectors: {str(e)}")
    
    @tool
    def ask_knowledgebase(self, query: str) -> str:
        """Query the knowledge base with the provided question."""
        try:
            print(f"DEBUG: Starting query for: {query}")
            
            # Step 1: Generate embedding for the query
            embedding = self._generate_embedding(query)
            
            # Step 2: Query the vector database
            vectors = self._query_vectors(embedding, top_k=5)
            
            # Step 3: Return simplified results
            if not vectors:
                self.last_documents = []  # Clear previous results
                return f"No documents found for: {query}"
            
            # Store structured documents for server access
            structured_documents = []
            results = []
            for i, vector in enumerate(vectors, 1):
                metadata = vector.get("metadata", {})
                
                # Extract and properly type the values
                distance = float(vector.get("distance", 0.0)) if vector.get("distance") != "unknown" else 0.0
                content = str(metadata.get("AMAZON_BEDROCK_TEXT", "No text available"))
                source = str(metadata.get("x-amz-bedrock-kb-source-uri", "Unknown source"))
                page_num = int(metadata.get("x-amz-bedrock-kb-document-page-number", 0))
                
                # Extract related content URIs as string list
                related_uris = []
                bedrock_metadata_str = metadata.get("AMAZON_BEDROCK_METADATA", "{}")
                if bedrock_metadata_str:
                    try:
                        bedrock_metadata = json.loads(bedrock_metadata_str)
                        related_contents = bedrock_metadata.get("relatedContents", [])
                        # Handle case where relatedContents might be None
                        if related_contents:
                            for content_item in related_contents:
                                if "s3Location" in content_item and "uri" in content_item["s3Location"]:
                                    related_uris.append(str(content_item["s3Location"]["uri"]))
                    except (json.JSONDecodeError, KeyError):
                        pass  # Skip if can't parse metadata
                
                # Create structured document
                document = {
                    "id": f"citation_{i}",
                    "distance": distance,
                    "source": source,
                    "page_number": page_num,
                    "related_uris": related_uris,
                    "content": content
                }
                
                # Store for server access
                structured_documents.append(document)
                
                results.append(f"document_{i}: {json.dumps(document, ensure_ascii=False, indent=2)}")
                results.append("")
            
            # Store the documents for server access
            self.last_documents = structured_documents
            
            final_result = "\n".join(results)
            
            # Debug: Show exactly what the LLM receives
            print("=" * 60)
            print("EXACT CONTEXT SENT TO LLM:")
            print("=" * 60)
            print(final_result)
            print("=" * 60)
            print("END OF LLM CONTEXT")
            print("=" * 60)
            
            return final_result
            
        except Exception as e:
            print(f"ERROR in ask_knowledgebase: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error querying knowledge base: {str(e)}"
    
    def get_last_documents_and_clear(self) -> List[Dict[str, Any]]:
        """Get the last retrieved documents and clear the cache.
        
        Returns:
            List of document dictionaries from the last query
        """
        documents = self.last_documents.copy()
        self.last_documents = []  # Clear after retrieval
        return documents


# Create a global instance for easy use
_kb_query = KnowledgeBaseQuery()

# Export the tool function for direct use
ask_knowledgebase = _kb_query.ask_knowledgebase

# Export function to get documents
def get_last_kb_documents_and_clear() -> List[Dict[str, Any]]:
    """Get the last knowledge base documents and clear them."""
    return _kb_query.get_last_documents_and_clear()


