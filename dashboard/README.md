# Hybrid Healthcare Advisory Dashboard

React + Tailwind dashboard for the **Hybrid Healthcare Advisory System**.

## Setup

```bash
cd dashboard
npm install
```

## Configure API

Env is loaded from the **repo root** (Vite `envDir` points to `..`). Copy `.env.example` → `.env` at the project root:

```bash
API_KEY=
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The dashboard uploads images to: `POST ${VITE_API_BASE_URL}/predict/severity`

If `VITE_API_BASE_URL` is not set, the frontend defaults to `http://127.0.0.1:8000`.

Primary expected response (your current FastAPI response):

```json
{
  "filename": "xray.png",
  "clinical_metrics": {
    "pneumonia_probability": "79.4%",
    "severity_index": 79.4,
    "status": "High Risk / Critical"
  },
  "rag_advisory": "## Clinical Advisory\n...",
  "disclaimer": "This is an AI-generated advisory..."
}
```

The UI also supports a fallback compact response:

```json
{ "is_pneumonia": true, "severity_score": 79.4, "advisory": "## Clinical Advisory\n..." }
```

## Run

```bash
npm run dev
```

