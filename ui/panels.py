def gamma_bar_html(g: dict | None, price: float | None = None, desk: str = "") -> str:
    """The regime every desk trades inside: dealer gamma from the Deribit chain, with its key levels."""
    if not g or not g.get("available"):
        why = (g or {}).get("note") or "options chain unavailable"
        return (f"<div class='ti-gamma neutral'><i class='ti-gamma-mark'></i><span class='ti-gamma-reg'>GAMMA REGIME UNKNOWN</span>"
                f"<span class='ti-gamma-play'>{html.escape(why)}</span></div>")
    label, tone, play = GAMMA_READ.get(g.get("regime"), ("GAMMA NEUTRAL", "neutral", "Net dealer gamma is flat at spot."))
    zgl = g.get("zero_gamma_level")
    side = ""
    if zgl and price:
        side = f" · price {'above' if price >= zgl else 'below'} the flip by {fmt_num(abs(price - zgl), 0, '$')}"
    net = g.get("net_gex_usd_per_1pct")
    net_txt = "—" if net is None else f"{'+' if net >= 0 else '-'}${abs(net) / 1e6:,.1f}M"
    levels = [("Put wall", g.get("put_wall")), ("Zero gamma", zgl), ("Max gamma", g.get("max_gamma_strike")),
              ("Call wall", g.get("call_wall"))]
    lv = "".join(f"<span><b>{n}</b> {fmt_num(v, 0, '$')}</span>" for n, v in levels)
    n_exp = len(g.get("expiries") or [])
    days_raw = g.get("horizon_days")
    days = float(days_raw) if days_raw is not None else 30.0
    scope = (f"{desk}: options settling in the next " + ("24 hours" if days == 1 else f"{days:g} days")) if desk else ""
    src_txt = f" ({g['source']})" if g.get("source") else ""
    tip = (f"Net GEX {net_txt} per 1% move, from {g.get('options_used', 0)} options across {n_exp} expiries in the "
           f"next {'24 hours' if days == 1 else f'{days:g} days'}{src_txt}. Assumes the naive dealer sign: calls positive, puts negative. "
           "Real dealer positioning is not published.")
    return (f"<div class='ti-gamma {tone}' title='{html.escape(tip)}'><i class='ti-gamma-mark'></i>"
            f"<span class='ti-gamma-reg'>{label}</span>"
            + (f"<span class='ti-gamma-scope'>{html.escape(scope)}</span>" if scope else "") +
            f"<span class='ti-gamma-net'>net {net_txt} / 1%{html.escape(side)}</span>"
            f"<span class='ti-gamma-lv'>{lv}</span>"
            f"<span class='ti-gamma-play'>{html.escape(play + _wall_sides(g, price))}</span></div>")
