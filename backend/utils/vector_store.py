"""Vector store implementation using ChromaDB for RAG functionality."""

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path
import uuid

from ..config import settings
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


class DesignVectorStore:
    """Vector store for UI/UX design patterns and references."""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.vector_db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="design_patterns",
                metadata={"description": "UI/UX design patterns and references"}
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_design_pattern(
        self,
        pattern_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        description: str = ""
    ) -> bool:
        """Add a design pattern to the vector store."""
        try:
            self.collection.add(
                ids=[pattern_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                documents=[description]
            )
            logger.debug(f"Added design pattern: {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add design pattern {pattern_id}: {e}")
            return False
    
    def search_similar_patterns(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar design patterns."""
        try:
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": n_results
            }
            
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            results = self.collection.query(**query_params)
            
            # Format results
            similar_patterns = []
            for i in range(len(results["ids"][0])):
                pattern = {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i],
                    "description": results["documents"][0][i]
                }
                similar_patterns.append(pattern)
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Failed to search similar patterns: {e}")
            return []
    
    def search_by_text_query(
        self,
        text_query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search patterns using text query."""
        try:
            # Encode text query to embedding
            query_embedding = self.embeddings.encode_text(text_query)
            
            return self.search_similar_patterns(
                query_embedding=query_embedding,
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to search by text query: {e}")
            return []
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific pattern by ID."""
        try:
            results = self.collection.get(ids=[pattern_id])
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "metadata": results["metadatas"][0],
                    "description": results["documents"][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pattern {pattern_id}: {e}")
            return None
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern from the vector store."""
        try:
            self.collection.delete(ids=[pattern_id])
            logger.debug(f"Deleted pattern: {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_patterns": count,
                "collection_name": self.collection.name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_patterns": 0, "collection_name": "unknown"}
    
    def populate_sample_data(self):
        """Populate the vector store with sample UI/UX patterns."""
        sample_patterns = [
            {
                "id": "navigation_hamburger_menu",
                "description": "Hamburger menu navigation pattern for mobile interfaces",
                "metadata": {
                    "category": "navigation",
                    "platform": "mobile",
                    "complexity": "simple",
                    "usability_score": 8.5
                }
            },
            {
                "id": "card_based_layout",
                "description": "Card-based layout for content organization and visual hierarchy",
                "metadata": {
                    "category": "layout",
                    "platform": "web",
                    "complexity": "medium",
                    "usability_score": 9.0
                }
            },
            {
                "id": "floating_action_button",
                "description": "Floating action button for primary actions in material design",
                "metadata": {
                    "category": "interaction",
                    "platform": "mobile",
                    "complexity": "simple",
                    "usability_score": 8.0
                }
            },
            {
                "id": "breadcrumb_navigation",
                "description": "Breadcrumb navigation for hierarchical content structure",
                "metadata": {
                    "category": "navigation",
                    "platform": "web",
                    "complexity": "simple",
                    "usability_score": 7.5
                }
            },
            {
                "id": "progressive_disclosure",
                "description": "Progressive disclosure pattern to reduce cognitive load",
                "metadata": {
                    "category": "information_architecture",
                    "platform": "both",
                    "complexity": "medium",
                    "usability_score": 8.8
                }
            }
        ]
        
        for pattern in sample_patterns:
            try:
                # Generate embedding for the description
                embedding = self.embeddings.encode_text(pattern["description"])
                
                self.add_design_pattern(
                    pattern_id=pattern["id"],
                    embedding=embedding,
                    metadata=pattern["metadata"],
                    description=pattern["description"]
                )
                
            except Exception as e:
                logger.warning(f"Failed to add sample pattern {pattern['id']}: {e}")


# Global vector store instance
_vector_store_instance: Optional[DesignVectorStore] = None


def get_vector_store() -> DesignVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = DesignVectorStore()
    return _vector_store_instance
