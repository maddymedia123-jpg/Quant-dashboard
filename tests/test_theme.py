from ui.theme import card_html, chip, fmt_num, fmt_pct, kpi_html, palette, tone_for_direction


def test_formatters_handle_none_and_numbers():
    assert fmt_num(None) == "—"
    assert fmt_num(79200.456, 0, "$") == "$79,200"
    assert fmt_num(0.1234, 2, suffix="%") == "0.12%"
    assert fmt_pct(1.234) == "+1.23%"
    assert fmt_pct(-0.5, 1) == "-0.5%"
    assert fmt_pct(None) == "—"


def test_kpi_html_escapes_and_renders_all_items():
    h = kpi_html([{"label": "Spot <b>", "value": "$1", "delta": "+1%", "tone": "up"}, {"label": "OI", "value": "—"}])
    assert "&lt;b&gt;" in h and h.count('class="ti-kpi"') == 2 and 'class="d up"' in h


def test_card_and_chip_and_palette():
    assert 'class="ti-card down"' in card_html("Bias", "<p>x</p>", "down")
    assert chip("Kraken", "up").startswith('<span class="ti-chip up">')
    assert palette(True)["bg"] != palette(False)["bg"]
    assert tone_for_direction("BULLISH") == "up" and tone_for_direction("NEUTRAL") == "neutral"
