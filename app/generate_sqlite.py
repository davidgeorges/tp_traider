import pandas as pd
import sqlite3

df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "profit_dqn": [0.0, 1.4, 2.5],
    "profit_baseline": [0.0, 0.3, 0.9]
})

conn = sqlite3.connect("profits.db")
df.to_sql("profits", conn, index=False, if_exists="replace")
conn.close()