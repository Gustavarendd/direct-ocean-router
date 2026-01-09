from __future__ import annotations

from pathlib import Path

from ocean_router.tss.tss_strtree import TSSSegmentIndex


def test_nearest_segment_and_penalty() -> None:
    fixture = Path(__file__).parent / "fixtures" / "tss_lane.geojson"
    index = TSSSegmentIndex.from_geojson(fixture)

    hit = index.nearest_segment(0.5, 0.05, max_dist_nm=10.0)
    assert hit is not None
    assert hit.allowed_dir == 1

    aligned = index.penalty_for_move(0.0, 0.0, 0.1, 0.0)
    misaligned = index.penalty_for_move(0.0, 0.0, 0.0, 0.1)
    assert aligned < misaligned
