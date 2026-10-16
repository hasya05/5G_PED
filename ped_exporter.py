"""Synthetic educational Prometheus exporter; not a real network model."""
import random
import time
from prometheus_client import Gauge, Counter, start_http_server

signal = Gauge("ped_signal_strength_dbm", "Simulated serving cell signal strength in dBm")
neighbor = Gauge("ped_neighbor_signal_strength_dbm", "Simulated neighbor cell signal strength in dBm")
speed = Gauge("ped_user_speed_kmh", "Simulated user speed in km/h")
load = Gauge("ped_cell_load_percent", "Simulated serving cell load percentage")
score_metric = Gauge("ped_handover_probability", "Illustrative handover score from 0 to 1")
decision = Gauge("ped_handover_recommended", "1 if illustrative policy recommends handover")
signal_contrib = Gauge("ped_explanation_signal_contribution", "Illustrative signal contribution")
neighbor_contrib = Gauge("ped_explanation_neighbor_contribution", "Illustrative neighbor contribution")
load_contrib = Gauge("ped_explanation_load_contribution", "Illustrative load contribution")
speed_contrib = Gauge("ped_explanation_speed_contribution", "Illustrative speed contribution")
recommendations = Counter("ped_handover_recommendations_total", "Simulated handover recommendations")

s, n, v, l = -78.0, -83.0, 34.0, 55.0
start_http_server(8000)
while True:
    s = max(-115, min(-55, s + random.uniform(-3.5, 2.5)))
    n = max(-115, min(-55, n + random.uniform(-3, 3)))
    v = max(0, min(140, v + random.uniform(-8, 8)))
    l = max(5, min(100, l + random.uniform(-7, 7)))
    sr = max(0.0, min(1.0, (-s - 70) / 35))
    na = max(0.0, min(1.0, (n - s + 12) / 24))
    lp = l / 100
    vf = min(1.0, v / 120)
    score = max(0.0, min(1.0, 0.45 * sr + 0.30 * na + 0.15 * lp + 0.10 * vf))
    rec = score >= 0.55 and n > s - 3
    signal.set(s); neighbor.set(n); speed.set(v); load.set(l)
    score_metric.set(score); decision.set(1 if rec else 0)
    signal_contrib.set(0.45 * sr); neighbor_contrib.set(0.30 * na)
    load_contrib.set(0.15 * lp); speed_contrib.set(0.10 * vf)
    if rec:
        recommendations.inc()
    time.sleep(5)
