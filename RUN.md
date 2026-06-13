# Running PawCare+ (React + FastAPI)

The project is split into two independent apps: a React **frontend** and a Python
**backend**. The frontend talks to a FastAPI service that wraps the existing
LangGraph workflow. **No business logic changed** — the API is a thin bridge over
`graph.assess_pet_health()`.

```
Hackthon/
├── frontend/            # React + TypeScript + Vite + Tailwind (UI)
├── backend/             # Python: LangGraph workflow, ML models, FastAPI API
│   ├── api/             #   FastAPI bridge (server.py, schemas.py)
│   ├── agents/ nodes/   #   16 agents · 17 graph nodes
│   ├── ml/ workflow/    #   sklearn models · routing metadata
│   ├── graph.py  state.py          # LangGraph core + workflow state
│   └── requirements.txt
├── .env.example         # template — copy to .env (never commit real keys)
└── README.md  RUN.md
```

```
┌────────────────────┐      POST /api/assess       ┌──────────────────────────┐
│  React (Vite/TS)   │ ───────────────────────────▶│  FastAPI (api/server.py)  │
│  frontend/ :5173   │ ◀─────────────────────────── │  → graph.assess_pet_health│
└────────────────────┘   JSON {result, summary}     │  → 17-node LangGraph + ML │
                                                     └──────────────────────────┘
```

## Quickest start — Docker

```bash
cp .env.example .env          # set OPENAI_API_KEY
docker compose up --build     # frontend → http://localhost:8080
```

See the **🐳 Run with Docker** section in `README.md` for details. The manual
(non-Docker) steps below are for local development.

## 1. Backend (FastAPI)

```bash
# from repo root — copy the env template and set your key
cp .env.example .env          # then edit .env: set OPENAI_API_KEY

cd backend
pip install -r requirements.txt          # adds fastapi + uvicorn (+ pins sklearn)
python -m uvicorn api.server:app --reload --port 8000
```

- Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## 2. Frontend (React)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api → :8000)
```

Build for production:

```bash
npm run build      # type-checks then emits frontend/dist/
npm run preview    # serve the production build locally
```

### Frontend env (`frontend/.env`, optional)

| Variable                  | Default                    | Purpose                                   |
| ------------------------- | -------------------------- | ----------------------------------------- |
| `VITE_API_BASE_URL`     | _(empty → same-origin)_ | Absolute API base for production hosting. |
| `VITE_API_PROXY_TARGET` | `http://localhost:8000`  | Where the dev server proxies `/api`.    |

## Security notes

- `.env` is now git-ignored. The real keys were previously **not** ignored — rotate
  them before pushing this repo anywhere, and keep using `.env.example` as the template.
- The bundled scikit-learn `.pkl` models (`backend/ml/models/`) were trained on
  1.8.x; `backend/requirements.txt` pins `scikit-learn>=1.8,<1.9` to avoid
  `InconsistentVersionWarning` and silent prediction drift. Re-train
  (`backend/ml/train_pipeline.py`) if you upgrade.
