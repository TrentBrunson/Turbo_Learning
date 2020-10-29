#%%
# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from config import DB_String
import pandas as pd

#%%
engine = create_engine(DB_String)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

#%%
# We can view all of the classes that automap found
Base.classes.keys()

#%%
# Create our session (link) from Python to the DB
session = Session(engine)

#%%
df = pd.read_sql_table('tex_combined_final', engine)
df.tail()

#%%
df = df.set_index("playid", drop = True)
df.index.name = "playID"

# %%
df.dtypes
# %%
df['clock'] = pd.to_datetime(df['clock'], format='%H:%M:%S').dt.time

# %%
