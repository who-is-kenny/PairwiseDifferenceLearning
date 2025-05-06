import sqlite3
import json

DATABASE_FILE = "PDL(RFC).db"
OUTPUT_JSON_FILE = "PDL(RFC).json"

def round_nested_list(data, precision=4):
    """
    Recursively rounds numerical values in nested lists.
    """
    if isinstance(data, list):
        return [round_nested_list(item, precision) for item in data]
    elif isinstance(data, (int, float)):
        return round(data, precision)
    return data  # Return the item as is if it's not a number or list

def export_db_to_json():
    """
    Exports the SQLite database contents to a JSON file with optimizations to reduce file size.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Query all rows from the compute_results table
    cursor.execute("SELECT model_name, pdc_perturbation, uncertainty, vmax, X, y, X_grid, kde_class0, mean_class0 FROM compute_results")
    rows = cursor.fetchall()

    # Get column names
    column_names = [description[0] for description in cursor.description]

    # Convert rows to a list of dictionaries
    results = []
    for row in rows:
        row_dict = dict(zip(column_names, row))
        # Deserialize JSON fields (proba, uncertainty, vmax, X, y, X_grid, kde_class0, mean_class0)
        # row_dict['proba'] = json.loads(row_dict['proba'])
        row_dict['uncertainty'] = json.loads(row_dict['uncertainty'])
        row_dict['vmax'] = json.loads(row_dict['vmax'])
        row_dict['X'] = json.loads(row_dict['X'])
        row_dict['y'] = json.loads(row_dict['y'])
        row_dict['X_grid'] = json.loads(row_dict['X_grid'])
        row_dict['kde_class0'] = json.loads(row_dict['kde_class0'])
        row_dict['mean_class0'] = json.loads(row_dict['mean_class0'])

        # Reduce precision of numerical data (to reduce file size)
        # row_dict['proba'] = round_nested_list(row_dict['proba'])
        row_dict['uncertainty'] = {key: [round(float(value), 4) for value in values] for key, values in row_dict['uncertainty'].items()}
        row_dict['vmax'] = {key: round(float(value), 4) for key, value in row_dict['vmax'].items()}
        row_dict['X'] = round_nested_list(row_dict['X'])
        row_dict['y'] = [round(float(value), 4) for value in row_dict['y']]
        row_dict['X_grid'] = round_nested_list(row_dict['X_grid'])
        row_dict['kde_class0'] = round_nested_list(row_dict['kde_class0'])
        row_dict['mean_class0'] = [round(float(value), 4) for value in row_dict['mean_class0']]

        results.append(row_dict)

    conn.close()

    # Write the results to a minified JSON file
    with open(OUTPUT_JSON_FILE, "w") as json_file:
        json.dump(results, json_file, separators=(',', ':'))  # Minify JSON by removing indentation

    print(f"Database exported to {OUTPUT_JSON_FILE}")

# Run the export function to convert current db file to JSON.
export_db_to_json()