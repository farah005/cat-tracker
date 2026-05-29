# 🐱 CatTracker — GPS Tracking & ML Prediction for Domestic Cats

> **PFA Demonstration Project** — Architecture moderne pour le suivi GPS d'un chat domestique avec prédiction de trajectoire par réseau LSTM.

---

## Table des matières

1. [Architecture](#architecture)
2. [Stack technique & justifications](#stack-technique--justifications)
3. [Installation rapide](#installation-rapide)
4. [Lancement complet](#lancement-complet)
5. [Chargement des données](#chargement-des-données)
6. [API Reference](#api-reference)
7. [Modèle LSTM](#modèle-lstm)
8. [Tests](#tests)
9. [Structure du projet](#structure-du-projet)
10. [Variables d'environnement](#variables-denvironnement)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        docker-compose                           │
│                                                                 │
│  ┌──────────────┐    REST/JSON    ┌──────────────────────────┐  │
│  │   Streamlit  │ ◄────────────► │      FastAPI Backend      │  │
│  │  (frontend)  │                │                           │  │
│  │  :8501       │                │  /cats        CRUD chats  │  │
│  │              │                │  /positions   spatial     │  │
│  │  • Carte     │                │  /upload      CSV ingest  │  │
│  │    Folium    │                │  /predict     LSTM        │  │
│  │  • Heatmap   │                │  :8000                    │  │
│  │  • Plotly    │                └──────────┬───────────────┘  │
│  └──────────────┘                           │ SQLAlchemy        │
│                                             │ + GeoAlchemy2     │
│                                  ┌──────────▼───────────────┐  │
│                                  │   PostgreSQL + PostGIS    │  │
│                                  │   :5432                   │  │
│                                  │                           │  │
│                                  │  chats      positions     │  │
│                                  │  (GEOMETRY, spatial idx)  │  │
│                                  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Flux de données :**
```
Collier GPS → CSV → POST /upload/{id}
                      │
                      ├─ Validation colonnes
                      ├─ Filtre médian (bruit GPS)
                      ├─ Calcul vitesse & distance maison
                      ├─ INSERT bulk PostgreSQL/PostGIS
                      └─ Background: réentraînement LSTM
                                          │
                              GET /predict/{id}
                                          │
                              [lat_norm, lon_norm,
                               hour_sin, hour_cos,
                               dist_home_norm, speed_norm]
                                          │
                                    LSTM (2 couches)
                                          │
                                  → lat_prédit, lon_prédit
```

---

## Stack technique & justifications

| Composant | Technologie | Justification |
|-----------|------------|---------------|
| **Base de données** | PostgreSQL 15 + PostGIS 3.3 | Standard industriel pour données géospatiales. PostGIS offre ST_Distance, ST_ConvexHull, index GIST – 10× plus rapide que du Python pur pour les requêtes spatiales. |
| **ORM** | SQLAlchemy 2 + GeoAlchemy2 | Type-safe, supporte les types géométriques PostGIS nativement, migrations faciles avec Alembic. |
| **Backend API** | FastAPI | Async natif, validation Pydantic automatique, OpenAPI/Swagger généré, performances comparables à Node.js. |
| **ML** | Keras/TensorFlow LSTM | Les séries temporelles GPS ont une forte dépendance temporelle. Un LSTM (Long Short-Term Memory) capture ces corrélations à long terme mieux qu'un MLP classique. |
| **Normalisation** | MinMaxScaler (sklearn) | Indispensable avant LSTM ; les coordonnées brutes (48.x, 2.x) ont des échelles différentes de hour_sin/cos ([-1,1]) |
| **Frontend** | Streamlit + Folium | Prototype rapide pour démonstration PFA. React serait plus scalable mais Streamlit suffit pour la démo et évite le JavaScript. |
| **Carte** | Folium (Leaflet.js) | Intégration Python native, support GeoJSON, heatmap, markers. |
| **Conteneurisation** | Docker Compose | Reproductibilité totale de l'environnement. Un `docker compose up` suffit. |
| **Tests** | pytest | Standard Python, fixtures, parametrize. |

### Pourquoi LSTM et pas un autre modèle ?

- **Random Forest / XGBoost** : Ne capturent pas l'ordre temporel des séquences.
- **ARIMA** : Univarié, mal adapté aux séries multivariées (lat+lon+heure+vitesse).
- **LSTM** : Conçu pour les séquences. La "mémoire" des cellules LSTM permet de se souvenir qu'un chat sort souvent au même endroit à la même heure.

**Architecture du réseau :**
```
Input  [batch, 6 steps, 6 features]
  ↓
LSTM(64 units, return_sequences=True)
  ↓
Dropout(0.2)
  ↓
LSTM(32 units)
  ↓
Dropout(0.2)
  ↓
Dense(16, relu)
  ↓
Dense(2)   # [lat_norm, lon_norm]
  ↓
MinMaxScaler.inverse_transform → [lat_WGS84, lon_WGS84]
```

---

## Installation rapide

### Prérequis

- **Docker** ≥ 24.0 + **Docker Compose** ≥ 2.24
- **Python** ≥ 3.10 (pour les scripts locaux uniquement)
- **make** (optionnel mais pratique)

```bash
# Cloner le projet
git clone https://github.com/yourname/cat-tracker.git
cd cat-tracker

# Copier et (optionnellement) éditer les variables d'environnement
cp .env.example .env
```

> **Personnaliser la position de la maison** : éditez `LAT_HOME` et `LON_HOME` dans `.env`.

---

## Lancement complet

### Option A — avec `make` (recommandé)

```bash
# Démarrer toute la stack
make up

# Générer + charger des données synthétiques (30 jours)
make seed-db

# Ouvrir le dashboard
open http://localhost:8501
```

### Option B — manuellement

```bash
# 1. Démarrer les services
docker compose up -d --build

# 2. Vérifier que tout est up
docker compose ps

# 3. Générer un jeu de données synthétique
python scripts/generate_synthetic_data.py --days 30 --output data/synthetic_cat.csv

# 4. Uploader le CSV via curl
curl -X POST "http://localhost:8000/upload/1" \
     -F "file=@data/synthetic_cat.csv"

# 5. Ouvrir le dashboard
open http://localhost:8501
# API Swagger : http://localhost:8000/docs
```

---

## Chargement des données

### Format CSV attendu

```csv
timestamp,latitude,longitude
2025-05-01 00:00:00,48.8566,2.3522
2025-05-01 00:10:00,48.8570,2.3525
2025-05-01 00:20:00,48.8568,2.3519
```

- `timestamp` : format ISO 8601 ou `YYYY-MM-DD HH:MM:SS` (UTC recommandé)
- `latitude`, `longitude` : degrés décimaux WGS-84
- Colonnes supplémentaires sont ignorées

### Traitement automatique lors de l'upload

1. Validation des colonnes obligatoires
2. **Filtre médian** (fenêtre=5) pour atténuer les pics GPS
3. Calcul de la vitesse entre points consécutifs (m/s)
4. Calcul de la distance à la maison (m) par Haversine
5. Insertion bulk avec `ON CONFLICT DO NOTHING` (idempotent)
6. **Réentraînement LSTM** en arrière-plan (si ≥ 56 points)

---

## API Reference

Base URL : `http://localhost:8000`

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Santé du service |
| `GET` | `/cats/` | Liste tous les chats |
| `POST` | `/cats/` | Créer un nouveau chat |
| `GET` | `/cats/{id}` | Détail d'un chat |
| `DELETE` | `/cats/{id}` | Supprimer un chat |
| `GET` | `/positions/{chat_id}` | Positions GPS (paginé) |
| `GET` | `/positions/{chat_id}/home-range` | Domaine vital (convex hull) |
| `POST` | `/upload/{chat_id}` | Uploader un CSV GPS |
| `GET` | `/predict/{chat_id}` | Prédire la prochaine position |

Documentation interactive : **http://localhost:8000/docs**

### Exemple — créer un chat

```bash
curl -X POST http://localhost:8000/cats/ \
  -H "Content-Type: application/json" \
  -d '{"nom":"Félix","race":"Maine Coon","lat_home":48.8566,"lon_home":2.3522}'
```

### Exemple — prédiction

```bash
curl http://localhost:8000/predict/1
# → {"chat_id":1,"predicted_latitude":48.8574,"predicted_longitude":2.3531,"model_version":"lstm_v1"}
```

---

## Modèle LSTM

### Features d'entrée (par pas de temps)

| Feature | Description | Normalisation |
|---------|-------------|---------------|
| `latitude` | Latitude WGS-84 | MinMax [0,1] |
| `longitude` | Longitude WGS-84 | MinMax [0,1] |
| `hour_sin` | sin(2π × heure/24) | Déjà [-1,1] |
| `hour_cos` | cos(2π × heure/24) | Déjà [-1,1] |
| `distance_home_m` | Distance Haversine à la maison | MinMax [0,1] |
| `vitesse_ms` | Vitesse entre deux fixes GPS | MinMax [0,1] |

**Encodage circulaire de l'heure** : sin/cos évite la discontinuité 23h→0h qui pénalise les modèles naïfs.

### Hyperparamètres

| Paramètre | Valeur | Variable d'env |
|-----------|--------|----------------|
| Longueur de séquence | 6 | `SEQUENCE_LEN` |
| Epochs max | 50 | `LSTM_EPOCHS` |
| Batch size | 32 | `LSTM_BATCH_SIZE` |
| Early stopping patience | 8 | — |
| Validation split | 10 % | — |

### Persistance

Les modèles sont sauvegardés dans `/app/ml_models/<chat_id>/` :
- `model.keras` — poids du réseau
- `scaler_in.pkl` — scaler des features
- `scaler_out.pkl` — scaler des sorties

---

## Tests

```bash
# Tests unitaires (pas de DB requise)
make test
# ou
pytest tests/ -v
```

**Couverture des tests :**

| Classe | Tests |
|--------|-------|
| `TestHaversine` | 6 cas (zéro, Paris↔Londres, symétrie, ~100m, positif, équateur) |
| `TestSpeedMs` | 4 cas (zéro dist, zéro temps, valeur connue, non-négatif) |
| `TestMedianFilter` | 4 cas (longueur, spike, signal plat, point unique) |
| `TestConvexHull` | 6 cas (GeoJSON, aire positive, aire correcte, centroïde, erreur <3 pts, grand dataset) |
| `TestPredictionSchema` | 3 cas (validation Pydantic des schémas de réponse) |

---

## Structure du projet

```
cat-tracker/
├── docker-compose.yml          # Orchestration 3 services
├── .env.example                # Template variables d'environnement
├── Makefile                    # Commandes pratiques
├── pytest.ini                  # Config pytest
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI entry point
│       ├── config.py           # Settings (pydantic-settings)
│       ├── database.py         # SQLAlchemy engine + session
│       ├── api/
│       │   ├── cats.py         # CRUD chats
│       │   ├── positions.py    # GPS positions + home range
│       │   └── upload.py       # CSV upload + prédiction
│       ├── models/
│       │   ├── orm.py          # SQLAlchemy ORM (Chat, Position)
│       │   └── schemas.py      # Pydantic I/O schemas
│       ├── services/
│       │   ├── geo.py          # Haversine, filtre médian, convex hull
│       │   └── ingestion.py    # Pipeline CSV → DB
│       └── ml/
│           └── lstm_predictor.py  # LSTM fit / predict / persist
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                  # Dashboard Streamlit
│
├── db/
│   └── init.sql                # PostGIS schema + triggers
│
├── scripts/
│   └── generate_synthetic_data.py  # Générateur de données réalistes
│
├── tests/
│   ├── conftest.py
│   └── test_geo.py             # 23 tests unitaires
│
└── data/                       # CSV générés (gitignored)
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `POSTGRES_USER` | `cattracker` | Utilisateur PostgreSQL |
| `POSTGRES_PASSWORD` | `cattracker_secret` | Mot de passe PostgreSQL |
| `POSTGRES_DB` | `cattracker_db` | Nom de la base |
| `POSTGRES_HOST` | `db` | Hôte PostgreSQL (nom service Docker) |
| `DATABASE_URL` | *(calculé)* | URL complète SQLAlchemy |
| `LAT_HOME` | `48.8566` | Latitude de la maison |
| `LON_HOME` | `2.3522` | Longitude de la maison |
| `BACKEND_URL` | `http://backend:8000` | URL backend (vue du frontend) |
| `SEQUENCE_LEN` | `6` | Fenêtre d'entrée LSTM |
| `LSTM_EPOCHS` | `50` | Epochs max d'entraînement |
| `LSTM_BATCH_SIZE` | `32` | Taille de batch |

---

## Requêtes spatiales PostGIS (exemples)

```sql
-- Distance entre deux points (en mètres)
SELECT ST_Distance(
    ST_Transform(geom, 3857),
    ST_Transform(ST_SetSRID(ST_MakePoint(2.3522, 48.8566), 4326), 3857)
) AS dist_m
FROM positions WHERE chat_id = 1
ORDER BY ts DESC LIMIT 1;

-- Convex hull (domaine vital) en SQL pur
SELECT ST_AsGeoJSON(ST_ConvexHull(ST_Collect(geom))) AS hull
FROM positions WHERE chat_id = 1;

-- Points dans un rayon de 500 m de la maison
SELECT COUNT(*) FROM positions
WHERE chat_id = 1
  AND ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(2.3522, 48.8566), 4326)::geography,
    500
  );
```

---

## Arrêt et nettoyage

```bash
# Arrêter la stack (garde les données)
make down

# Tout supprimer (volumes inclus)
make clean
```

---

*Projet réalisé dans le cadre d'un PFA — Architecture microservices Python avec ML embarqué.*
