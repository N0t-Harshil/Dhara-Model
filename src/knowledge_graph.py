import os
import json
import logging
from typing import List, Dict, Any, Tuple

import networkx as nx
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class GraphMemory:
    """
    Dynamic Knowledge Graph for Agentic Memory.
    Uses ChromaDB for semantic node search and NetworkX for relationship traversal.
    """
    def __init__(self, db_path: str = "./memory_vault"):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize Vector DB (Chroma)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(db_path, "chroma"))
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="concept_nodes",
            embedding_function=self.sentence_transformer_ef
        )
        
        # Initialize Graph DB (NetworkX)
        self.graph_file = os.path.join(db_path, "knowledge_graph.json")
        self.graph = nx.DiGraph()
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info(f"Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()

    def _save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(self.graph_file, "w") as f:
            json.dump(data, f)

    def add_interaction(self, problem: str, code: str, concepts: List[str], relationships: List[Tuple[str, str, str]]):
        """
        Store a successful interaction in the memory vault.
        concepts: ['FastAPI', 'JWT Auth', 'Login Endpoint']
        relationships: [('FastAPI', 'uses', 'JWT Auth'), ('Login Endpoint', 'part of', 'FastAPI')]
        """
        interaction_id = f"interaction_{self.collection.count()}"
        
        # 1. Add interaction context to Vector DB
        document = f"Problem: {problem}\nSolution:\n{code}"
        self.collection.add(
            documents=[document],
            metadatas=[{"type": "interaction", "concepts": ",".join(concepts)}],
            ids=[interaction_id]
        )
        
        # 2. Add nodes and edges to Graph
        self.graph.add_node(interaction_id, type="interaction", problem=problem, code=code)
        
        for concept in concepts:
            self.graph.add_node(concept, type="concept")
            self.graph.add_edge(interaction_id, concept, relation="involves")
            
        for source, rel, target in relationships:
            self.graph.add_node(source, type="concept")
            self.graph.add_node(target, type="concept")
            self.graph.add_edge(source, target, relation=rel)
            
        self._save_graph()
        logger.info(f"Added interaction {interaction_id} to memory vault.")

    def retrieve_context(self, query: str, top_k_nodes: int = 2) -> str:
        """
        Query the memory vault using GraphRAG.
        1. Semantic search for starting nodes.
        2. Traverse graph to find related context.
        """
        if self.collection.count() == 0:
            return "No previous memory available."

        # 1. Semantic Search in Vector DB
        n_results = min(top_k_nodes, self.collection.count())
        if n_results <= 0:
            return "No relevant memory found."
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        
        if not results['ids'] or not results['ids'][0]:
            return "No relevant memory found."

        retrieved_ids = results['ids'][0]
        documents = results['documents'][0]
        
        # 2. Graph Traversal (Pulling related concepts)
        context_blocks = []
        for i, node_id in enumerate(retrieved_ids):
            block = f"--- Past Memory {i+1} ---\n{documents[i]}\n"
            
            # Find related concepts in the graph
            if node_id in self.graph:
                neighbors = list(self.graph.neighbors(node_id))
                if neighbors:
                    block += "Related Concepts Built Previously: " + ", ".join(neighbors) + "\n"
                    
                    # 1-hop relationships
                    relations = []
                    for n in neighbors:
                        for target in self.graph.successors(n):
                            if self.graph.nodes[target].get("type") == "concept":
                                rel = self.graph.edges[n, target].get("relation", "related to")
                                relations.append(f"{n} -> {rel} -> {target}")
                    if relations:
                        block += "Architectural Rules: " + " | ".join(relations) + "\n"
                        
            context_blocks.append(block)

        return "\n".join(context_blocks)
