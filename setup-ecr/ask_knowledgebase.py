"""
Simple knowledge base query tool that retrieves documents from S3 vectors.
"""
import boto3
import json
import os
import unicodedata
from typing import Optional, List, Dict, Any  # Optional kept for potential re-enable of filter
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
        NOTE: filter capability disabled (was previously an optional metadata filter).
        """
        try:
            # Build dynamic filter based on environment selections set by server
            topic = os.getenv("KB_TOPIC") or None
            active_only = os.getenv("KB_ACTIVE_ONLY") == "1"

            filter_obj = None
            if topic and active_only:
                filter_obj = {"$and": [{"topics": topic}, {"stop_date": "none"}]}
            elif topic:
                filter_obj = {"topics": topic}
            elif active_only:
                filter_obj = {"stop_date": "none"}

            query_params = {
                "vectorBucketName": "acrelec-model-eval",
                "indexName": "acrelec-model-eval-index",
                "queryVector": {"float32": embedding},
                "topK": top_k,
                "returnDistance": True,
                "returnMetadata": True,
            }
            if filter_obj:
                query_params["filter"] = filter_obj
            
            response = self.s3vectors.query_vectors(**query_params)
            vectors = response.get("vectors", [])
            # Attach filter for transparency (non-standard field) by monkey patching attribute
            for v in vectors:
                v["_applied_filter"] = filter_obj
            return vectors
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

            # Capture requested topic for diagnostics / fallback filtering
            requested_topic = os.getenv("KB_TOPIC") or None

            def _norm(s: str) -> str:
                if s is None:
                    return ""
                nfkd = unicodedata.normalize("NFKD", str(s))
                return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

            def _topic_matches(meta_value, requested: str) -> bool:
                if not meta_value or not requested:
                    return False
                rn = _norm(requested)
                if isinstance(meta_value, (list, tuple, set)):
                    return any(rn in _norm(v) for v in meta_value if v is not None)
                return rn in _norm(meta_value)

            topic_fallback_summary: Dict[str, Any] = {}
            if requested_topic and vectors:
                # Determine whether any vector metadata already clearly matches (indicating filter likely worked)
                meta_matches = [v for v in vectors if _topic_matches(v.get("metadata", {}).get("topics"), requested_topic)]
                if not meta_matches and os.getenv("KB_TOPIC_FALLBACK", "1") == "1":
                    # Attempt client-side filtering
                    filtered = [v for v in vectors if _topic_matches(v.get("metadata", {}).get("topics"), requested_topic)]
                    if filtered:
                        topic_fallback_summary = {
                            "requested_topic": requested_topic,
                            "original_retrieved": len(vectors),
                            "kept_after_fallback": len(filtered),
                            "strategy": "accent-insensitive substring",
                            "note": "Applied client-side fallback filter because no upstream match detected"
                        }
                        vectors = filtered
                    else:
                        topic_fallback_summary = {
                            "requested_topic": requested_topic,
                            "original_retrieved": len(vectors),
                            "kept_after_fallback": len(vectors),
                            "strategy": "accent-insensitive substring",
                            "note": "No documents matched fallback; returning originals to avoid empty result"
                        }
                else:
                    topic_fallback_summary = {
                        "requested_topic": requested_topic,
                        "original_retrieved": len(vectors),
                        "upstream_topic_match_detected": bool(meta_matches),
                        "note": "Upstream filter appears to have worked or matches already present"
                    }
            
            # Step 3: Format and return the results
            if not vectors:
                transparency = {
                    "query": query,
                    "retrieved_count": 0,
                    "documents": [],
                    "applied_filter": None
                }
                return "No relevant documents found in the knowledge base.\n\n```json\n" + json.dumps(transparency, ensure_ascii=False, indent=2) + "\n```"
            
            results = []
            results.append(f"Found {len(vectors)} relevant documents for query: '{query}'\n")
            if requested_topic:
                results.append(f"Requested topic (KB_TOPIC): {requested_topic}")
            if topic_fallback_summary:
                results.append(f"Topic filter diagnostics: {json.dumps(topic_fallback_summary, ensure_ascii=False)}")
            
            debug_enabled = os.getenv("KB_DEBUG") == "1"
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
                        # Always include topics now for transparency
                        if key == "topics":
                            results.append(f"Topics: {value}")
                            doc_meta_output[key] = value
                            continue
                        results.append(f"{key.title()}: {value}")
                        doc_meta_output[key] = value

                structured_docs.append({
                    "file_id": clean_file_id,
                    "relevance_score": relevance_str,
                    "distance": distance,
                    "metadata": doc_meta_output
                })

                if debug_enabled:
                    # Include raw vector keys (excluding potentially large float arrays)
                    vector_copy = {k: v for k, v in vector.items() if k not in {"values", "vector", "float32"}}
                    results.append(f"[DEBUG] Raw Vector JSON: {json.dumps(vector_copy, default=str)[:1000]}")

                results.append("-" * 50)
            
            # Attempt to recover filter from first vector (if present)
            applied_filter = None
            if vectors and "_applied_filter" in vectors[0]:
                applied_filter = vectors[0]["_applied_filter"]

            transparency = {
                "query": query,
                "retrieved_count": len(structured_docs),
                "documents": structured_docs,
                "applied_filter": applied_filter,
                "topic_fallback": topic_fallback_summary if topic_fallback_summary else None
            }

            return "\n".join(results) + "\n\n```json\n" + json.dumps(transparency, ensure_ascii=False, indent=2) + "\n```"
            
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"


# Create a global instance for easy use
_kb_query = KnowledgeBaseQuery()

# Export the tool function for direct use
ask_knowledgebase = _kb_query.ask_knowledgebase


