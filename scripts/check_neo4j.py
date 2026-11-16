"""Smoke test: connect to Neo4j with the project settings and print its version.

Run with `make check-neo4j` or `python scripts/check_neo4j.py`.
"""

import sys

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from paper_search.config import settings


def main() -> int:
    """Print the Neo4j version, or a hint about what went wrong."""
    print(f"Connecting to {settings.neo4j_uri} as {settings.neo4j_user!r} ...")
    try:
        with GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        ) as driver:
            driver.verify_connectivity()
            records, _, _ = driver.execute_query(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            )
    except AuthError:
        print("ERROR: authentication failed. Check NEO4J_USER / NEO4J_PASSWORD in .env.")
        return 1
    except ServiceUnavailable:
        print(
            f"ERROR: no Neo4j at {settings.neo4j_uri}. Start it with `make neo4j-up`, "
            "then wait a few seconds for it to finish booting."
        )
        return 1
    except Neo4jError as exc:
        print(f"ERROR: Neo4j rejected the query: {exc}")
        return 1

    for record in records:
        print(f"OK: {record['name']} {record['version']} ({record['edition']} edition)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
