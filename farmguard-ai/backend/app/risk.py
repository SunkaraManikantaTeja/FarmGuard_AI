"""Transparent rule-based ESTIMATES. Not validated forecasts. Replace with a trained model when data exists."""
def estimate_risk(temp_c, humidity, rain_days_30, growth_stage="vegetative", prior_disease=False):
    s = 15 + max(0, humidity - 60) * 0.6 + rain_days_30 * 1.2
    if 20 <= temp_c <= 30: s += 10
    if growth_stage in ("tillering", "flowering"): s += 8
    if prior_disease: s += 10
    s = max(5, min(95, s))
    return {"estimated_risk_pct": round(s), "label": "estimate, not a confirmed disease",
            "advice": ["Scout the field regularly", "Improve drainage and airflow", "Consult a local agriculture officer before spraying"]}

def simulate(spray_interval, scouting_per_week, rainy_days):
    def r(iv, sc, hu): return max(5, min(95, 20 + hu * 1.6 + iv * 1.1 - sc * 4))
    cur, rec = r(spray_interval, scouting_per_week, rainy_days), r(10, 4, rainy_days)
    cost = lambda iv, sc: round(900 + (30 / iv) * 350 + sc * 60)
    return {"disclaimer": "Toy model with assumed coefficients. Not a guarantee.",
            "current": {"risk_pct": cur, "cost_per_acre": cost(spray_interval, 0), "loss_exposure_pct": round(cur * .6)},
            "recommended": {"risk_pct": rec, "cost_per_acre": cost(10, 4), "loss_exposure_pct": round(rec * .6)}}
