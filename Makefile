.PHONY: help up down build logs test generate-data seed-db clean

## ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  CatTracker – available commands"
	@echo "  ────────────────────────────────────────────────────"
	@echo "  make up            Start all services (docker-compose)"
	@echo "  make down          Stop all services"
	@echo "  make build         (Re)build images"
	@echo "  make logs          Tail all logs"
	@echo "  make test          Run unit tests (local Python)"
	@echo "  make generate-data Generate 30-day synthetic CSV"
	@echo "  make seed-db       Generate + upload CSV to running stack"
	@echo "  make clean         Remove volumes and images"
	@echo ""

## ── Docker ───────────────────────────────────────────────────────────────────

up:
	cp -n .env.example .env 2>/dev/null || true
	docker compose up -d --build
	@echo ""
	@echo "  ✅  Stack is up."
	@echo "  Frontend : http://localhost:8501"
	@echo "  API docs : http://localhost:8000/docs"
	@echo ""

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	cd backend && pip install -q -r requirements.txt && cd ..
	pytest tests/ -v

## ── Data helpers ─────────────────────────────────────────────────────────────

generate-data:
	python scripts/generate_synthetic_data.py \
		--days 30 \
		--output data/synthetic_cat.csv
	@echo "CSV written to data/synthetic_cat.csv"

seed-db: generate-data
	@echo "Uploading CSV to running backend (cat id=1) …"
	curl -s -X POST "http://localhost:8000/upload/1" \
		-F "file=@data/synthetic_cat.csv" | python3 -m json.tool
	@echo ""
	@echo "✅  Seed complete – open http://localhost:8501 to explore."

## ── Clean ────────────────────────────────────────────────────────────────────

clean:
	docker compose down -v --rmi local
	rm -rf data/synthetic_cat.csv
