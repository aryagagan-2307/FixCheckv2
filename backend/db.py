import os
from pathlib import Path
from typing import Optional

from pydantic import NaiveDatetime
from sqlmodel import Field, SQLModel, create_engine

# Everything the app stores lives in one folder, so resetting is easy.
DATA = Path(os.getenv("FIXCHECK_DATA", Path(__file__).parent / "data"))
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DATA / 'fixcheck.db'}", connect_args={"check_same_thread": False})


class Complaint(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    category: str
    ward: str = ""
    lat: float
    lon: float
    near_school: bool = False
    # open | reopened | closed_pending | closed
    status: str = "open"
    created_at: NaiveDatetime       # server time, set once, never reset
    photo: str
    photo_hash: str
    confidence: float = 0.0
    severity: float = 0.0
    duplicate_count: int = 0
    reopen_count: int = 0
    dispute_count: int = 0
    model_mismatch: bool = False    # last closure failed a model-based check
    measures: str = "{}"            # JSON from the photo models: summary, signals, coverage...


class Attempt(SQLModel, table=True):
    """One officer closure attempt with proof."""
    id: Optional[int] = Field(default=None, primary_key=True)
    complaint_id: int = Field(index=True)
    at: NaiveDatetime
    photo: str
    photo_hash: str
    lat: float
    lon: float
    passed: bool
    failures: str = "[]"            # JSON list of {code, text}
    warnings: str = "[]"


class Event(SQLModel, table=True):
    """Timeline entries shown on the tracker page."""
    id: Optional[int] = Field(default=None, primary_key=True)
    complaint_id: int = Field(index=True)
    at: NaiveDatetime
    kind: str
    text: str
    photo: Optional[str] = None


def init_db():
    SQLModel.metadata.create_all(engine)
    # databases made before the `measures` column existed: add it instead of crashing
    with engine.begin() as con:
        cols = [r[1] for r in con.exec_driver_sql("PRAGMA table_info(complaint)").fetchall()]
        if "measures" not in cols:
            con.exec_driver_sql("ALTER TABLE complaint ADD COLUMN measures VARCHAR DEFAULT '{}'")
