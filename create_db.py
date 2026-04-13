"""
the structure: 3-30-300.duckdb: schemas - raw, rules, results

- raw => preprocessing => rules => results

= logic by scripts: 
1.
- greenery_exploration.ipynb   (raw => preprocessing.bologna_parks)
- buildings_prep.ipynb         (raw => preprocessing.bologna_buildings)

2.
- 300m.ipynb                   (preprocessing + raw => rule_300m.bologna_results)
- 30pct.ipynb                  (preprocessing + raw => rule_30.bologna_results)
- 3trees.ipynb                 (preprocessing + raw => rule_3.bologna_results)
  
- results

"""

import duckdb
from config import DB_PATH

con = duckdb.connect(str(DB_PATH))
con.execute("INSTALL spatial; LOAD spatial;")

for schema in ["raw", "rule_300m", "rule_30", "rule_3", "results"]:
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")

print("DB initialized:", DB_PATH)
con.close()

