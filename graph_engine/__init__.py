"""
graph_engine — lightweight knowledge-graph extraction package.

Sits alongside the existing GraphRAG / LanceDB pipeline in dev-server.
Both pipelines can run on the same collection for side-by-side comparison.

Public surface
--------------
  from graph_engine.models import Triple, Node, Edge
  from graph_engine.extractors.hybrid_extractor import HybridExtractor
  from graph_engine.store import GraphStore
  from graph_engine.pipeline import GraphPipeline
"""
from graph_engine.models import Edge, Node, Triple
from graph_engine.pipeline import GraphPipeline

__all__ = ["Node", "Edge", "Triple", "GraphPipeline"]
