# Care Document Analyzer

A full-stack portfolio prototype that turns **synthetic** care notes and incident reports into structured, searchable records.

## Stack
- React + TypeScript (Vite)
- FastAPI + Pydantic
- PostgreSQL (Docker Compose)
- Anthropic Claude tool calling (optional; demo mode is default)
- Automated labeled evaluation script

## Run locally
1. Copy `.env.example` to `.env`.
2. Run `docker compose up --build` from this folder.
3. Open http://localhost:8000/docs for API docs.
4. In another terminal: `cd frontend && npm install && npm run dev`.
5. Open http://localhost:5173.

Use only fabricated data. This prototype is not a medical device and does not provide clinical advice. Review every extraction. Do not claim accuracy metrics until the evaluation script has actually been run.
