import os
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

import emoji
import pandas as pd
import sqlite3
import streamlit as st

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY2')
print(OPENAI_API_KEY)
TABLES = ['Album', 'Artist', 'Track']

QUERIES = ("How many albums are there?"
    , "Which album has the most tracks?"
    , "What artist has the album with the most tracks?"
)

st.title(emoji.emojize(":robot_face: Text to SQL via LangChain :robot_face:"))
st.subheader("table metadata")

def _get_metadata():

    con = sqlite3.connect("Chinook.db")
    cur = con.cursor()

    ts, cs = list(), list()
    for table in TABLES:

        rows = cur.execute("select * from %s limit 1" % table)
        cols = [k[0] for k in rows.description]

        ts.append(table)
        cs.append(cols)

    con.close()
    return pd.DataFrame({'table': ts, 'columns': cs})

st.write(_get_metadata())

llm = OpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY)
db = SQLDatabase.from_uri("sqlite:///Chinook.db", include_tables=TABLES)
db_chain = SQLDatabaseChain.from_llm(llm, db, use_query_checker=True, return_intermediate_steps=True)

for i, query in enumerate(QUERIES):

    output = db_chain(query)
    sql, result = output["intermediate_steps"][1], output["result"]

    st.subheader("query %s" % str(1 + i))

    st.write("Q: %s" % query)
    st.code(sql)
    st.write("A: %s" % result)