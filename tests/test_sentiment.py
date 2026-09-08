import json
import pathlib

from core.data.sentiment import parse_fng

FX = pathlib.Path(__file__).parent / "fixtures"


def test_parse_fng():
    s = parse_fng(json.loads((FX / "fng.json").read_text()))
    assert s.available and s.source == "alternative.me"
    assert 0 <= s.fear_greed <= 100
    assert s.classification
    assert 0 <= s.fear_greed_prev <= 100
