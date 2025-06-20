-- one table to register each cluster
CREATE TABLE clusters (
  cluster_id   SERIAL PRIMARY KEY,
  created_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
  summary      TEXT         NULL       -- optional cluster-level summary
);

-- one table to store every message, tagged by cluster
CREATE TABLE messages (
  message_id   SERIAL PRIMARY KEY,
  cluster_id   INT          NOT NULL REFERENCES clusters(cluster_id),
  timestamp    TIMESTAMP    NOT NULL,
  sender       TEXT         NOT NULL,
  content      TEXT         NOT NULL,
  embedding    BYTEA        NOT NULL    -- or JSON/ARRAY of floats
);