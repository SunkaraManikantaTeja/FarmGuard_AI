import sqlite3, os
DB = os.path.join(os.path.dirname(__file__), "..", "data", "farmguard.db")  # swap for PostgreSQL later via SQLAlchemy

def conn():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT DEFAULT CURRENT_TIMESTAMP,
        crop TEXT, language TEXT, mode TEXT, prediction TEXT, confidence REAL)""")
    return c

def save(crop, lang, mode, pred, conf):
    with conn() as c:
        c.execute("INSERT INTO predictions(crop,language,mode,prediction,confidence) VALUES(?,?,?,?,?)",
                  (crop, lang, mode, pred, conf))

def history(limit=50):
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]
