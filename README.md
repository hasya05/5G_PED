# 5G PED Monitoring Framework — Ready to Upload

This bundle includes a **simple browser dashboard that does not require Docker**. It is the recommended route if you want to get a working public demo online without repeating the Docker/Codespaces setup issues.

## Option A — Put the dashboard online (recommended, no Docker)

1. Download and unzip this project.
2. Create a new **public or private GitHub repository**.
3. Upload all files and folders from this unzipped folder to the repository (not the ZIP itself).
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
5. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
6. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

The app is interactive: use **Generate new sample** in the sidebar to simulate changing 5G conditions, view the illustrative handover score and explanation, inspect recent samples, and download the history as CSV.

## Option B — Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Important project note

The included Streamlit dashboard is a self-contained fallback for a quick, dependable demo. It does **not** require Prometheus, Grafana, Docker, or a Docker daemon. Its scores and explanations are synthetic educational examples, not outputs from a trained model or real SHAP.

The `docker-compose.yml`, `Dockerfile`, `ped_exporter.py`, and `prometheus.yml` files are also included for the original Prometheus/Grafana stack. That stack requires a host where Docker Compose works; it is not required for Option A.

## Original Prometheus + Grafana stack (optional)

On a machine with Docker Engine and Docker Compose available:

```bash
docker compose up --build -d
```

- Grafana: `http://localhost:3000` — demo login `admin` / `ped-demo-change-this`
- Prometheus: `http://localhost:9090`
- Metrics: `http://localhost:8000/metrics`

Change the demo Grafana password before sharing any deployed instance. Never expose the demo credentials on a public service.
