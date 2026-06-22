import pandas as pd
import sqlite3 as sql
import os

def create_database():
    # check existence of bd before creation of new one
    if os.path.exists("../shopping_sessions.db"):
        os.remove("../shopping_sessions.db")

    # create connection with database
    db = sql.connect("../shopping_sessions.db")
    cursor = db.cursor()

    print("", "=" * 60, "\n1. DATABASE STRUCTURE CREATION...\n", "=" * 60)

    # create normalized tables
    create_tables_sql = """
        CREATE TABLE IF NOT EXISTS sessions(
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT,
            weekend BOOLEAN,
            special_day INTEGER,
            os INTEGER,
            browser INTEGER,
            traffic_type INTEGER,
            region INTEGER,
            visitor_type_id REAL,
            revenue BOOLEAN NOT NULL,
            FOREIGN KEY (visitor_type_id) REFERENCES visitor_types(visitor_type_id)
        );

        CREATE TABLE IF NOT EXISTS sessions_metrics(
            session_id INTEGER PRIMARY KEY,
            administrative INTEGER,
            administrative_duration REAL,
            informational INTEGER,
            informational_duration REAL,
            product INTEGER,
            product_duration REAL,
            bounce_rates REAL NOT NULL,
            exit_rates REAL NOT NULL,
            page_values REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS visitor_types(
            visitor_type_id INTEGER PRIMARY KEY,
            visitor_type TEXT NOT NULL
        );
    """

    # tables creation
    cursor.executescript(create_tables_sql)
    print("✓ TABLES CREATED SUCCESSFULLY")

    print("", "=" * 60, "\n2. FILLING OUT DIRECTORIES...\n", "=" * 60)

    # fill out directory
    visitor_types_directory = [
        (1, 'Returning_Visitor'),
        (2, 'New_Visitor'),
        (3, 'Other'),
    ]

    cursor.executemany("INSERT OR IGNORE INTO visitor_types (visitor_type_id, visitor_type) Values (?, ?)",
                       visitor_types_directory)

    print(f"✓ DIRECTORY visitor_types WAS FILLED WITH {len(visitor_types_directory)} ENTRIES")

    print("", "=" * 60, "\n3. DATA LOADING AND PROCESSING...\n", "=" * 60)
    df = pd.read_csv('../data/online_shoppers_intention.csv')

    # fill out sessions table
    def get_placeholders(row_data):
        return ', '.join(['?' for _ in range(len(row_data))])

    try:
        # create mapper for visitor type
        cursor.execute("SELECT * from visitor_types")
        visitor_type_mapping = {v_type: v_id for v_id, v_type in cursor.fetchall()}

        sessions_data = []
        row_final = ()

        for _, row in df.iterrows():
            row_final = (
                row['Month'],
                row['Weekend'],
                row['SpecialDay'],
                row['OperatingSystems'],
                row['Browser'],
                row['TrafficType'],
                row['Region'],
                visitor_type_mapping[row['VisitorType']],
                row['Revenue']
            )
            sessions_data.append(row_final)

        placeholders = get_placeholders(row_final)
        cursor.executemany(f"""
            INSERT INTO sessions (month, weekend, special_day, os, browser, traffic_type, region, visitor_type_id, revenue)
            VALUES ({placeholders})""", sessions_data)

        print(f"✓ SESSIONS LOADED: {len(sessions_data)}")

    except Exception as e:
        print(f"✗ Error during sessions data loading: {e}")

    # fill out sessions_metrics table
    try:
        metrics_data = []
        row_final = ()

        for _, row in df.iterrows():
            row_final = (
                row['Administrative'],
                row['Administrative_Duration'],
                row['Informational'],
                row['Informational_Duration'],
                row['ProductRelated'],
                row['ProductRelated_Duration'],
                row['BounceRates'],
                row['ExitRates'],
                row['PageValues']
            )
            metrics_data.append(row_final)

        placeholders = get_placeholders(row_final)
        cursor.executemany(f"""
            INSERT INTO sessions_metrics (administrative, administrative_duration, informational, informational_duration, product, product_duration, bounce_rates, exit_rates, page_values)
            VALUES ({placeholders})""", metrics_data)

        print(f"✓ SESSIONS METRICS LOADED: {len(metrics_data)}")

    except Exception as e:
        print(f"✗ Error during sessions metrics data loading: {e}")

    print("", "=" * 60, "\n4. INDEXES CREATION FOR OPTIMIZATION...\n", "=" * 60)

    # create indexes
    indexes_sql = """
        CREATE INDEX IF NOT EXISTS idx_month ON sessions (month);
        CREATE INDEX IF NOT EXISTS idx_os ON sessions (os);
        CREATE INDEX IF NOT EXISTS idx_browse ON sessions (browser);
        CREATE INDEX IF NOT EXISTS idx_visitor_types_id ON sessions (visitor_type_id);
        CREATE INDEX IF NOT EXISTS idx_administrative_duration ON sessions_metrics (administrative_duration);
        CREATE INDEX IF NOT EXISTS idx_informational_duration ON sessions_metrics (informational_duration);
        CREATE INDEX IF NOT EXISTS idx_product_duration ON sessions_metrics (product_duration);
    """

    cursor.executescript(indexes_sql)

    print("✓ INDEXES WERE CREATED")

    print("", "=" * 60, "\n5. DATABASE SAVING AND CLOSING...\n", "=" * 60)

    # save all changes
    db.commit()

    # close connection with database
    db.close()

    print('✓ DATABASE WAS CREATED SUCCESSFULLY AND IS READY TO USE')


def database_validation():
    db = sql.connect("../shopping_sessions.db")
    cursor = db.cursor()

    print("", "=" * 60, "\n6. DATA INTEGRITY CHECK...\n", "=" * 60)

    # create check queries
    percent_calc = "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sessions), 2)"

    check_queries = {
        "Total quantity of sessions": "SELECT COUNT(*) FROM sessions",
        "Total metrics for sessions": "SELECT COUNT(*) FROM sessions_metrics",
        "Total success sessions in %": f"SELECT {percent_calc} AS revenue_percent FROM sessions WHERE revenue = 1",
        "Total sessions without purchases in %": f"SELECT {percent_calc} AS failure_percent FROM sessions  WHERE revenue = 0",
        "Total returning visitors in %": f"SELECT {percent_calc} AS returning_visitors_percent FROM sessions  WHERE visitor_type_id = 1",
        "Total new visitors in %": f"SELECT {percent_calc} AS new_visitors_percent FROM sessions  WHERE visitor_type_id = 2",
        "Purchases by visitor types": """
            SELECT vp.visitor_type,
            COUNT(*) AS purchases,
            ROUND(
                100.0 * COUNT(*) / (SELECT COUNT(*) FROM sessions WHERE revenue = 1),
                2
            ) as percent
            FROM sessions s
            JOIN visitor_types vp ON s.visitor_type_id = vp.visitor_type_id
            WHERE s.revenue = 1
            GROUP BY vp.visitor_type
        """,
    }

    print("SOME STATISTICS FROM DB:")

    for name, query in check_queries.items():
        cursor.execute(query)
        result = cursor.fetchall()
        print(f"{name}:")
        for row in result:
            if len(row) == 1:
                print(row[0])
            else:
                print(row)

    print("", "=" * 60, "\n7. ANALYTICAL QUERIES...\n", "=" * 60)

    # create some analytical queries
    analytical_queries = {
        "Top 3 months with purchases": """
            SELECT s.month, COUNT(*) AS purchases 
            FROM sessions s
            WHERE s.revenue = 1
            GROUP BY s.month
            ORDER BY purchases DESC
            LIMIT 3
        """,
        "Top 3 browsers": """
            SELECT s.browser, COUNT(*) AS session_counts
            FROM sessions s
            GROUP BY s.browser
            ORDER BY session_counts DESC
            LIMIT 3
        """,
        "Top 3 regions": """
            SELECT s.region, COUNT(*) AS session_counts
            FROM sessions s
            GROUP BY s.region
            ORDER BY session_counts DESC
            LIMIT 3
        """,
        "Top 3 traffic types": """
            SELECT s.traffic_type, COUNT(*) AS session_counts
            FROM sessions s
            GROUP BY s.traffic_type
            ORDER BY session_counts DESC
            LIMIT 3
        """,
    }

    for name, query in analytical_queries.items():
        print(f"{name}:")
        cursor.execute(query)
        columns = [descr[0] for descr in cursor.description]
        results = cursor.fetchall()

        for row in results:
            print("  " + " | ".join(f"{col}: {val}" for col, val in zip(columns, row)))


    # get final statistics
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = cursor.fetchall()

    print("CREATED TABLES IN DB:")
    for table in tables:
        name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        print(f"{name}: {count} entries")

    # close connection with database
    db.close()


def load_data_from_db():
    db = sql.connect("../shopping_sessions.db")

    query = """
        SELECT
            sm.administrative AS Administrative,
            sm.administrative_duration AS Administrative_Duration,
            sm.informational AS Informational,
            sm.informational_duration AS Informational_Duration,
            sm.product AS ProductRelated,
            sm.product_duration AS ProductRelated_Duration,
            sm.bounce_rates AS BounceRates,
            sm.exit_rates AS ExitRates,
            sm.page_values AS PageValues,
            
            s.special_day AS SpecialDay,
            s.month AS Month,
            s.os AS OperatingSystems,
            s.browser AS Browser,
            s.region AS Region,
            s.traffic_type AS TrafficType,
            
            vt.visitor_type AS VisitorType,
            
            s.weekend AS Weekend,
            s.revenue AS Revenue
            
            FROM sessions s
            INNER JOIN sessions_metrics sm ON s.session_id = sm.session_id
            INNER JOIN visitor_types vt ON s.visitor_type_id = vt.visitor_type_id 
            
    """

    df = pd.read_sql_query(query, con=db)

    db.close()

    return df