import os
import string
from datetime import datetime, date
from decimal import Decimal
import numpy as np
import psycopg2
from gensim.models import Word2Vec

def is_float_num(v):
    try:
        float(v)
        return True
    except Exception:
        return False

def to_float_maybe(x):
    """Return float value or None for non-convertible inputs."""
    if x is None:
        return None
    if isinstance(x, (int, float, Decimal)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None

def to_datetime_maybe(x):
    """Return datetime or None from date/datetime/str."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        # normalize pure date to datetime
        return datetime.combine(x, datetime.min.time())
    if isinstance(x, str):
        s = x.strip()
        # try a few common formats
        for fmt in ("%Y-%m-%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d",
                    "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
    return None

def postgresql_nodes_types():
    nodes = ['Seq Scan', 'Index Scan', 'Bitmap Index Scan', 'Bitmap Heap Scan', 'Index Only Scan', 'CTE Scan', 'Subquery Scan',
             'Hash', 'Hash Join', 'Merge Join', 'Nested Loop',
             'Sort', 'Incremental Sort', 'Aggregate', 'WindowAgg', 'Gather Merge', 'Gather', 'Group',
             'Unique', 'Memoize', 'Materialize', 'SetOp', 'Append', 'Merge Append', 'Result', 'Limit']
    return {node: index for index, node in enumerate(nodes)}

def postgresql_database_statistic_generator(db_params):
    # Get attributes of each table
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT pg_tables.tablename FROM pg_tables WHERE tablename NOT LIKE 'pg%' AND tablename NOT LIKE 'sql_%' ORDER BY tablename;")
    result = cursor.fetchall()
    tables = [table[0] for table in result]

    index = 0
    tables_index = {}
    tables_index_all = {}
    columns_list = []
    columns_index = {}
    table_columns_number = []
    attribute_range = {}

    numeric_types = {'integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint'}
    datetime_types = {'date', 'time without time zone', 'time with time zone',
                      'timestamp without time zone', 'timestamp with time zone'}
    text_types = {'character', 'character varying', 'text'}

    for i, table in enumerate(tables):
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
        result = cursor.fetchall()
        tables_index[table] = i
        tables_index_all[table] = i
        table_columns_number.append(len(result))

        for (column_name, data_type) in result:
            full_column_name = f"{table}.{column_name}"
            tables_index_all[full_column_name] = i
            columns_list.append(column_name)
            columns_index[full_column_name] = index
            # columns_index[column_name] = index
            index += 1

            cursor.execute(f"SELECT {full_column_name} FROM {table};")
            result_c = cursor.fetchall()

            # ---------- Numeric ----------
            if data_type in numeric_types:
                values = [to_float_maybe(v[0]) for v in result_c]
                filtered_values = [x for x in values if x is not None]
                if filtered_values:
                    value_set = set(filtered_values)
                    max_num = max(value_set)
                    min_num = min(value_set)
                    attribute_count = len(value_set)
                    attri_num = 1.0 / attribute_count if attribute_count else 0.0
                else:
                    min_num, max_num, attri_num = 0.0, 0.0, 0.0
                attribute_range[full_column_name] = [min_num, max_num, attri_num]

            # ---------- Date/Time ----------
            elif data_type in datetime_types:
                dt_vals = []
                for v in result_c:
                    dt = to_datetime_maybe(v[0])
                    if dt is not None:
                        dt_vals.append(dt)
                if not dt_vals:
                    # keep stable defaults but warn
                    print(f"{data_type} type warning: No valid date/time entries found for {full_column_name}.")
                    dt_vals = [datetime.min]
                dt_set = set(dt_vals)
                max_datetime = max(dt_set)
                min_datetime = min(dt_set)
                attri_num = 1.0 / len(dt_set) if dt_set else 0.0
                attribute_range[full_column_name] = [min_datetime, max_datetime, attri_num]

            # ---------- Text ----------
            elif data_type in text_types:
                original_values = [str(v[0]) for v in result_c if v[0] is not None]
                cleaned_values = []
                for value in original_values:
                    cleaned = value.strip().lower().translate(str.maketrans('', '', string.punctuation))
                    cleaned_values.extend(cleaned.split())
                unique_values = set(original_values)
                attri_num = 1.0 / len(unique_values) if unique_values else 0.0

                # Train/save word2vec only if we have tokens
                if cleaned_values:
                    model = Word2Vec(sentences=[cleaned_values], vector_size=1, window=5, min_count=1, workers=8)
                    model_dir = os.path.join('..', 'Data', db_params['dbname'], 'database_statistics', 'Word2vec')
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f'{table}_{column_name}.model')
                    model.save(model_path)
                else:
                    model_path = 'None'
                attribute_range[full_column_name] = [model_path, attri_num]

            else:
                print(f"Data type {data_type} not supported for column {full_column_name}.")
                # still ensure a placeholder to avoid KeyError later
                attribute_range[full_column_name] = [None, None, 0.0]

    save_path = os.path.join('..', 'Data', db_params['dbname'], 'database_statistics')
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "tables_index"), tables_index)
    np.save(os.path.join(save_path, "tables_index_all"), tables_index_all)
    np.save(os.path.join(save_path, "columns_index"), columns_index)
    np.save(os.path.join(save_path, "attribute_range"), attribute_range)
    np.save(os.path.join(save_path, "table_columns_number"), table_columns_number)
    np.save(os.path.join(save_path, "columns_list"), list(set(columns_list)))
    np.save(os.path.join(save_path, "postgresql_nodestypes_all"), postgresql_nodes_types())

def generate_postgresql_database_statistic(db_params):
    postgresql_database_statistic_generator(db_params)

if __name__ == '__main__':
    db_params = {
        "dbname": 'imdb',
        "user": 'postgres',
        "password": '123456',
        "host": 'localhost',
        "port": '5432'
        # "options": "-c statement_timeout=600000"
    }
    generate_postgresql_database_statistic(db_params)


