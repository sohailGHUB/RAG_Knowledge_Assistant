def rewrite_query(query):
    """
    Normalize and expand common technical queries
    with useful retrieval terms.
    """

    query = query.strip()

    if not query:
        return query

    query = query.replace("?", "")
    query = query.replace("!", "")
    query = " ".join(query.split())

    query_lower = query.lower()

    expansions = {
        "delta lake": (
            "Delta Lake transaction log ACID "
            "schema management time travel"
        ),
        "azure data factory": (
            "Azure Data Factory ADF pipelines "
            "data integration orchestration"
        ),
        "databricks": (
            "Azure Databricks lakehouse Spark "
            "Delta Lake notebooks compute"
        ),
        "synapse": (
            "Azure Synapse Analytics SQL "
            "data warehouse analytics"
        ),
        "spark": (
            "Apache Spark distributed processing "
            "PySpark DataFrame cluster"
        ),
        "sql database": (
            "Azure SQL Database relational database "
            "SQL queries tables"
        )
    }

    for phrase, expansion in expansions.items():

        if phrase in query_lower:

            return f"{query} {expansion}"

    return query