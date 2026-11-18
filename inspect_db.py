import sqlite3

conn = sqlite3.connect('chroma_multimodal_db/chroma.sqlite3')
cur = conn.cursor()

emb_cols = cur.execute("PRAGMA table_info(embeddings);").fetchall()
print('embeddings cols:', [c[1] for c in emb_cols])

meta_cols = cur.execute("PRAGMA table_info(embedding_metadata);").fetchall()
print('embedding_metadata cols:', [c[1] for c in meta_cols])

# Count how many embeddings belong to Ratna_Raju_Project_Details.pdf
cur.execute("""
    SELECT em.key, em.string_value, count(*) 
    FROM embedding_metadata em 
    WHERE em.string_value LIKE '%Ratna_Raju_Project_Details.pdf%'
    GROUP BY em.key, em.string_value;
""")
print('Metadata counts:', cur.fetchall())
