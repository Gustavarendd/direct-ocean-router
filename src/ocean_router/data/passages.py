"""Macro passage graph utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml
import networkx as nx


@dataclass
class PassageGraph:
    path: Path

    def __post_init__(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.graph = nx.DiGraph()
        for node in data.get("nodes", []):
            self.graph.add_node(node["id"], **node)
        for edge in data.get("edges", []):
            self.graph.add_edge(edge["from"], edge["to"], **edge)

    def macro_route(self, start_basin: str, end_basin: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.graph, start_basin, end_basin, weight="weight_nm")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def corridor_waypoints(self, path: List[str]) -> List[List[float]]:
        waypoints: List[List[float]] = []
        for i in range(len(path) - 1):
            edge = self.graph.edges[path[i], path[i + 1]]
            waypoints.extend(edge.get("waypoints", []))
        return waypoints
