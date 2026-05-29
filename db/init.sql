-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Cats table
CREATE TABLE IF NOT EXISTS chats (
    id          SERIAL PRIMARY KEY,
    nom         VARCHAR(100) NOT NULL,
    race        VARCHAR(100),
    couleur     VARCHAR(50),
    poids_kg    FLOAT,
    lat_home    FLOAT NOT NULL DEFAULT 48.8566,
    lon_home    FLOAT NOT NULL DEFAULT 2.3522,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Positions table with PostGIS geometry
CREATE TABLE IF NOT EXISTS positions (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL,
    latitude    FLOAT NOT NULL,
    longitude   FLOAT NOT NULL,
    geom        GEOMETRY(POINT, 4326),
    vitesse_ms  FLOAT,
    distance_home_m FLOAT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Spatial index for fast geo queries
CREATE INDEX IF NOT EXISTS idx_positions_geom    ON positions USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_positions_chat_ts ON positions (chat_id, ts DESC);

-- Auto-populate geometry column from lat/lon
CREATE OR REPLACE FUNCTION set_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom := ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_geom ON positions;
CREATE TRIGGER trg_set_geom
BEFORE INSERT OR UPDATE ON positions
FOR EACH ROW EXECUTE FUNCTION set_geom();

-- Insert a default cat for demo purposes
INSERT INTO chats (nom, race, couleur, poids_kg, lat_home, lon_home)
VALUES ('Mimi', 'Européen', 'tigré', 4.2, 48.8566, 2.3522)
ON CONFLICT DO NOTHING;
