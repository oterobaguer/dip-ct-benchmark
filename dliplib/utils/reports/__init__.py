import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_results_table(table, table_key):
    """
    Saves results (hyper param selection) to a file
    :param table: ResultTable
    :param table_key: Key that identifies the results
    """
    path = os.path.join(BASE_DIR, 'reports', table_key + '.txt')

    with open(path, "w") as f:
        f.write(table.to_string(show_columns=['misc'], max_colwidth=1000))
        f.close()
