"""Configuration loader and dataclasses for ocean router settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class AlgorithmConfig:
    """Algorithm selection configuration."""
    default: str = "auto"
    coarse_to_fine_threshold_nm: float = 100.0
    coarse_scale: int = 8


@dataclass
class CorridorConfig:
    """Corridor settings configuration."""
    width_short_nm: float = 100.0
    width_long_nm: float = 100.0
    long_route_threshold_nm: float = 1500.0
    offshore_buffer_nm: float = 500.0
    chokepoint_buffer_nm: float = 50.0
    min_width_nm: float = 25.0


@dataclass
class TSSConfig:
    """TSS (Traffic Separation Scheme) configuration."""
    wrong_way_penalty: float = 1000.0
    alignment_weight: float = 0.5
    lane_crossing_penalty: float = 1.5
    sepzone_crossing_penalty: float = 1000.0
    sepboundary_crossing_penalty: float = 10.0
    proximity_check_radius: int = 2
    max_lane_deviation_deg: float = 45.0


@dataclass
class LandConfig:
    """Land/coast configuration."""
    min_distance_nm: float = 2.0
    proximity_penalty: float = 5.0


@dataclass
class DepthConfig:
    """Depth/bathymetry configuration."""
    near_shore_penalty: float = 5.0


@dataclass
class SimplifyConfig:
    """Path simplification configuration."""
    tolerance_nm: float = 0.5
    max_simplify_nm: float = 100.0
    max_skip: int = 50
    min_land_distance_cells_default: int = 10
    min_land_distance_cells_tss: int = 3


@dataclass
class BypassConfig:
    """TSS bypass configuration."""
    offset_distances_nm: List[float] = field(default_factory=lambda: [5, 10, 15, 20, 30, 40, 50])


@dataclass
class RoutingConfig:
    """General routing configuration."""
    allow_diagonals: bool = True
    max_turn_deg: float = 90.0
    density_bias: float = -1.0


@dataclass
class RouterConfig:
    """Complete router configuration."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    tss: TSSConfig = field(default_factory=TSSConfig)
    land: LandConfig = field(default_factory=LandConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    bypass: BypassConfig = field(default_factory=BypassConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "RouterConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            algorithm=AlgorithmConfig(**data.get('algorithm', {})),
            corridor=CorridorConfig(**data.get('corridor', {})),
            tss=TSSConfig(**data.get('tss', {})),
            land=LandConfig(**data.get('land', {})),
            depth=DepthConfig(**data.get('depth', {})),
            simplify=SimplifyConfig(**data.get('simplify', {})),
            bypass=BypassConfig(**data.get('bypass', {})),
            routing=RoutingConfig(**data.get('routing', {})),
        )


# Global config instance - lazily loaded
_config: Optional[RouterConfig] = None


def get_config(config_path: Optional[Path] = None) -> RouterConfig:
    """Get the global configuration, loading from file if not already loaded.
    
    Args:
        config_path: Path to the config file. If None, uses default location.
        
    Returns:
        The RouterConfig instance.
    """
    global _config
    
    if _config is None or config_path is not None:
        if config_path is None:
            # Default to configs/routing_defaults.yaml relative to project root
            # Try to find it relative to this file's location
            this_file = Path(__file__)
            project_root = this_file.parent.parent.parent.parent
            config_path = project_root / "configs" / "routing_defaults.yaml"
        
        if config_path.exists():
            _config = RouterConfig.from_yaml(config_path)
        else:
            # Use defaults if config file not found
            _config = RouterConfig()
    
    return _config


def reload_config(config_path: Optional[Path] = None) -> RouterConfig:
    """Force reload of configuration from file.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        The newly loaded RouterConfig instance.
    """
    global _config
    _config = None
    return get_config(config_path)
