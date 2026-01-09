from shapely.geometry import Polygon

from ocean_router.tss.tss_graph import (
    TSSGraphConfig,
    build_directed_lane_graph,
    make_connectors,
    route_with_tss,
)


class DummyRouter:
    def route(self, start, goal):
        return [start, goal], {"tss_used": False}


def _lane_feature(coords, flow=90.0, forward=True):
    return {
        "type": "Feature",
        "properties": {
            "seamark:type": "separation_lane",
            "tss_flow_bearing_deg": flow,
            "tss_flow_is_forward_along_geometry": forward,
        },
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _zone_feature(coords):
    return {
        "type": "Feature",
        "properties": {"seamark:type": "separation_zone"},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def test_wrong_way_rejected():
    features = [_lane_feature([(0.0, 0.0), (1.0, 0.0)])]
    graph = build_directed_lane_graph(features, sepzone_buffer_nm=0.2)
    config = TSSGraphConfig(connector_radius_nm=5.0, max_connectors=4)
    path, meta = route_with_tss(
        start=(1.0, 0.0),
        goal=(0.0, 0.0),
        open_sea_router=DummyRouter(),
        tss_graph=graph,
        forbidden=[],
        config=config,
    )
    assert path == [(1.0, 0.0), (0.0, 0.0)]
    assert meta.get("tss_used") is False


def test_separation_zone_blocks_edges():
    lane = _lane_feature([(0.0, 0.0), (1.0, 0.0)])
    zone = _zone_feature([(-0.5, -0.1), (1.5, -0.1), (1.5, 0.1), (-0.5, 0.1), (-0.5, -0.1)])
    graph = build_directed_lane_graph([lane, zone], sepzone_buffer_nm=0.2)
    assert graph.edges == []
    assert len(graph.zones) == 1


def test_connectors_avoid_forbidden_polygons():
    features = [_lane_feature([(1.0, -1.0), (1.0, 1.0)])]
    graph = build_directed_lane_graph(features, sepzone_buffer_nm=0.2)
    forbidden = [Polygon([(0.4, -0.2), (0.6, -0.2), (0.6, 0.2), (0.4, 0.2)])]
    connectors = make_connectors(
        (0.0, 0.0),
        graph,
        forbidden,
        radius_nm=120.0,
        max_connectors=4,
        entry_angle_weight=1.0,
        max_angle_deg=120.0,
        crossing_target_deg=90.0,
        crossing_angle_weight=1.0,
        crossing_max_angle_deg=70.0,
        crossing_penalty=2.0,
    )
    assert connectors == []
