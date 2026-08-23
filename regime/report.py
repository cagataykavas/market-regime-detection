from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    stats_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{int(values['observations'])}</td>"
        f"<td>{values['mean_return']:.4%}</td>"
        f"<td>{values['volatility']:.4%}</td>"
        f"<td>{values['drawdown']:.2%}</td>"
        f"<td>{values['confidence']:.1%}</td>"
        "</tr>"
        for name, values in result["regime_stats"].items()
    )
    transitions = result["transition_matrix"]
    states = sorted(transitions)
    transition_header = "".join(f"<th>{html.escape(state)}</th>" for state in states)
    transition_rows = "".join(
        f"<tr><th>{html.escape(state)}</th>"
        + "".join(f"<td>{transitions[state].get(other, 0.0):.1%}</td>" for other in states)
        + "</tr>"
        for state in states
    )
    duration_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{days:.1f}</td></tr>"
        for name, days in result["average_regime_duration_days"].items()
    )
    evaluation = result.get("evaluation")
    eval_card = ""
    if evaluation:
        eval_card = (
            '<div class="card"><div class="big">'
            f"{evaluation['adjusted_rand_index']:.3f}</div><div>Synthetic ARI</div></div>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Regime Report</title><style>
body{{background:#09101d;color:#edf3ff;font-family:Inter,system-ui,sans-serif;margin:0;padding:32px}}main{{max-width:1120px;margin:auto}}.muted{{color:#96a7c0}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}.card{{background:#121c30;border:1px solid #293954;border-radius:14px;padding:18px}}.big{{font-size:27px;font-weight:800}}
table{{width:100%;border-collapse:collapse;background:#121c30;margin:12px 0 28px}}th,td{{padding:10px;border-bottom:1px solid #293954;text-align:left}}th{{color:#aec2ff}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr}}body{{padding:16px}}}}
</style></head><body><main><p class="muted">Unsupervised market-state reference project</p><h1>Market Regime Detection</h1>
<div class="cards"><div class="card"><div class="big">{result['rows']}</div><div>Price observations</div></div><div class="card"><div class="big">{len(result['regime_stats'])}</div><div>Named regimes</div></div>{eval_card}</div>
<h2>Regime statistics</h2><table><thead><tr><th>Regime</th><th>Obs.</th><th>Mean rolling return</th><th>Volatility</th><th>Drawdown</th><th>Mean confidence</th></tr></thead><tbody>{stats_rows}</tbody></table>
<h2>Transition matrix</h2><table><thead><tr><th>From / To</th>{transition_header}</tr></thead><tbody>{transition_rows}</tbody></table>
<h2>Average contiguous duration</h2><table><thead><tr><th>Regime</th><th>Business days</th></tr></thead><tbody>{duration_rows}</tbody></table>
<p class="muted">GMM observations are modeled as a mixture; the transition table is measured after classification rather than learned as an HMM transition process.</p>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
