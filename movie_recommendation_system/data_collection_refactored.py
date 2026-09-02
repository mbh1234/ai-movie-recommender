#!/usr/bin/env python3
"""
data_collection_refactored.py

Continuously collects Kafka log events if data files are missing,
until the user presses ENTER to stop, then builds:

  - interactions_table.csv
  - unique_users.csv
  - unique_movies.csv
  - cleaned movies_table.csv (if available)

If the interactions table already exists, the script exits early.

Notes:
  • Handles raw log lines from Kafka (non-JSON).
  • Automatically resumes data processing after stopping collection.
"""

import os
import json
import threading
import logging
from typing import List
import pandas as pd
from kafka import KafkaConsumer, TopicPartition
from tqdm import tqdm

# ---------------------------------------------------
# CONFIGURATION CLASS
# ---------------------------------------------------
class DataConfig:
    """Data loading and processing configuration"""
    # File paths
    users_file: str = "data/users.csv"
    movies_file: str = "data/movies_table_clean.csv"
    interactions_file: str = "data/interactions_table.csv"

    # Data validation
    min_interactions_per_user: int = 5
    min_interactions_per_movie: int = 10
    max_rating: float = 5.0
    min_rating: float = 1.0

    # Feature engineering
    watch_time_percentiles: List[float] = None

    def __post_init__(self):
        if self.watch_time_percentiles is None:
            self.watch_time_percentiles = [0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------
# LOGGING
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("data_pipeline")

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
OUT_DIR = "./outputs"
WATCH_FILE = os.path.join(OUT_DIR, "watch_events.jsonl")
RATING_FILE = os.path.join(OUT_DIR, "rating_events.jsonl")
REC_FILE = os.path.join(OUT_DIR, "recommendation_events.jsonl")
INTERACTIONS_FILE = os.path.join(OUT_DIR, "interactions_table.csv")
USERS_FILE = os.path.join(OUT_DIR, "unique_users.csv")
MOVIES_FILE = os.path.join(OUT_DIR, "unique_movies.csv")
MOVIES_TABLE = os.path.join(OUT_DIR, "movies.csv")
CLEANED_MOVIES_FILE = os.path.join(OUT_DIR, "movies_table_clean.csv")

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "movielog6")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# ---------------------------------------------------
# UTILITIES
# ---------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def iter_jsonl(path: str):
    """Iterate through lines of a JSONL file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line.strip())
            except Exception:
                continue

# ---------------------------------------------------
# KAFKA COLLECTION
# ---------------------------------------------------
def consume_from_kafka_until_user_stops():
    """
    Consume raw Kafka log lines until user presses ENTER.
    Stores them in JSONL files, classified by type.
    """
    ensure_dir(OUT_DIR)
    logger.info(f"Connecting to Kafka topic='{KAFKA_TOPIC}' @ {KAFKA_BOOTSTRAP} ...")

    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=None,
    )

    tp = TopicPartition(KAFKA_TOPIC, 0)
    consumer.assign([tp])

    stop_flag = {"stop": False}

    def wait_for_user():
        input("\n📡 Collecting Kafka events... Press ENTER anytime to stop.\n")
        stop_flag["stop"] = True
        logger.info("User requested to stop Kafka collection...")

    threading.Thread(target=wait_for_user, daemon=True).start()

    watch_f = open(WATCH_FILE, "a", encoding="utf-8")
    rating_f = open(RATING_FILE, "a", encoding="utf-8")
    rec_f = open(REC_FILE, "a", encoding="utf-8")

    begin = consumer.beginning_offsets([tp])[tp]
    end = consumer.end_offsets([tp])[tp]
    total = end - begin
    logger.info(f"Starting collection of ~{total} messages...")

    count = 0
    with tqdm(total=total) as pbar:
        for msg in consumer:
            if stop_flag["stop"]:
                break

            line = msg.value.decode("utf-8").strip()

            # classify event type
            if "recommendation request" in line.lower():
                rec_f.write(json.dumps({"raw": line}) + "\n")
            elif "get /data/m/" in line.lower():
                watch_f.write(json.dumps({"raw": line}) + "\n")
            elif "get /rate/" in line.lower():
                rating_f.write(json.dumps({"raw": line}) + "\n")
            else:
                watch_f.write(json.dumps({"raw": line}) + "\n")

            count += 1
            pbar.update(1)

            if msg.offset + 1 >= end:
                break

    watch_f.close()
    rating_f.close()
    rec_f.close()
    consumer.close()
    logger.info(f"Kafka collection stopped after {count} messages.")

# ---------------------------------------------------
# INTERACTIONS TABLE BUILDER
# ---------------------------------------------------
def build_interactions_table():
    """Aggregate raw events into an interactions table."""
    logger.info("Building interactions table...")

    stats = {}
    users = set()
    movies = set()

    for ev in iter_jsonl(WATCH_FILE):
        raw = ev.get("raw", "")
        parts = raw.split(",")
        if len(parts) < 3:
            continue
        timestamp, user, path = parts[0], parts[1], parts[2]
        movie = path.split("/data/m/")[-1] if "/data/m/" in path else None
        if not user or not movie:
            continue
        key = (str(user), str(movie))
        entry = stats.setdefault(key, {"watch_count": 0, "total_minutes": 0.0, "rating": None})
        entry["watch_count"] += 1
        users.add(str(user))
        movies.add(str(movie))

    for ev in iter_jsonl(RATING_FILE):
        raw = ev.get("raw", "")
        parts = raw.split(",")
        if len(parts) < 3:
            continue
        timestamp, user, path = parts[0], parts[1], parts[2]
        movie = path.split("/rate/")[-1] if "/rate/" in path else None
        if not user or not movie:
            continue
        key = (str(user), str(movie))
        entry = stats.setdefault(key, {"watch_count": 0, "total_minutes": 0.0, "rating": None})
        entry["rating"] = 5.0
        users.add(str(user))
        movies.add(str(movie))

    rows = [{"user_id": u, "movie_id": m, **vals} for (u, m), vals in stats.items()]
    df = pd.DataFrame(rows)

    ensure_dir(OUT_DIR)
    df.to_csv(INTERACTIONS_FILE, index=False)
    pd.DataFrame(sorted(users), columns=["user_id"]).to_csv(USERS_FILE, index=False)
    pd.DataFrame(sorted(movies), columns=["movie_id"]).to_csv(MOVIES_FILE, index=False)

    logger.info(f"Wrote {len(df)} interaction rows to {INTERACTIONS_FILE}")
    logger.info(f"Unique users: {len(users)}, movies: {len(movies)}")

# ---------------------------------------------------
# MOVIES TABLE CLEANER
# ---------------------------------------------------
def clean_movies_table():
    """Clean JSON-like fields in movies.csv."""
    if not os.path.exists(MOVIES_TABLE):
        logger.warning(f"Movies table not found: {MOVIES_TABLE}")
        return
    logger.info("Cleaning movies metadata table...")
    df = pd.read_csv(MOVIES_TABLE)
    for col in ["genres", "production_companies"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("'", '"')
    df.to_csv(CLEANED_MOVIES_FILE, index=False)
    logger.info(f"Saved cleaned movies table -> {CLEANED_MOVIES_FILE}")

# ---------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------
def main():
    cfg = DataConfig()

    # If interactions file exists, exit early
    if os.path.exists(cfg.interactions_file):
        logger.info(f"✅ Data already present at '{cfg.interactions_file}'. Exiting.")
        return

    ensure_dir(OUT_DIR)

    if not (os.path.exists(WATCH_FILE) and os.path.exists(RATING_FILE)):
        logger.info("Event files not found. Starting Kafka log collection...")
        consume_from_kafka_until_user_stops()
    else:
        logger.info("Event files found — skipping Kafka collection.")

    logger.info("Building datasets...")
    build_interactions_table()
    clean_movies_table()
    logger.info("✅ Data pipeline completed successfully.")

if __name__ == "__main__":
    main()
