import sqlite3
import json
import os
import pandas as pd
import numpy as np

DATABASE_FILE = "PDL(RFC).db"  # Change this for another database name if needed

def initialize_database():
    """
    Initializes the SQLite database and creates the necessary table if it doesn't exist.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compute_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            pdc_perturbation TEXT,
            uncertainty BLOB,
            vmax BLOB,
            X BLOB,
            y BLOB,
            X_grid BLOB,
            kde_class0 BLOB,
            mean_class0 BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_compute_results(model_name, pdc_perturbation, compute_results, X, y, X_grid):
    """
    Saves the computed results into the SQLite database.

    Parameters:
    - model_name: Name of the model used.
    - pdc_perturbation: Perturbation method used.
    - compute_results: Dictionary containing the computed results.
    - X: Training data features.
    - y: Training data labels.
    - X_grid: Grid data used for plotting or computation.
    """
    print(f"Saving compute results for model: {model_name}, perturbation: {pdc_perturbation}")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Serialize data to JSON
    # proba_json = json.dumps(compute_results['proba'].tolist())
    uncertainty_serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                for key, value in compute_results['uncertainty'].items()}
    uncertainty_json = json.dumps(uncertainty_serializable)
    vmax_json = json.dumps(compute_results['vmax'])
    X_json = json.dumps(X.tolist())
    y_json = json.dumps(y.tolist())
    X_grid_json = json.dumps(X_grid.tolist())
    kde_class0_json = json.dumps(compute_results['kde_class0'].tolist())
    mean_class0_json = json.dumps(compute_results['mean_class0'].tolist())

    cursor.execute("""
        INSERT INTO compute_results (model_name, pdc_perturbation, uncertainty, vmax, X, y, X_grid, kde_class0, mean_class0)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (model_name, pdc_perturbation, uncertainty_json, vmax_json, X_json, y_json, X_grid_json, kde_class0_json, mean_class0_json))

    conn.commit()
    conn.close()

def load_compute_results(model_name, pdc_perturbation):
    """
    Loads the computed results from the SQLite database.

    Parameters:
    - model_name: Name of the model used.
    - pdc_perturbation: Perturbation method used.

    Returns:
    - A dictionary containing the computed results, or None if no matching entry is found.
    """
    print(f"Loading compute results for model: {model_name}, perturbation: {pdc_perturbation}")
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT uncertainty, vmax, X, y, X_grid, kde_class0, mean_class0
        FROM compute_results
        WHERE model_name = ? AND pdc_perturbation = ?
    """, (model_name, pdc_perturbation))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    # Deserialize data from JSON
    # proba = np.array(json.loads(row[0]))
    uncertainty = json.loads(row[0])
    vmax = json.loads(row[1])
    X = np.array(json.loads(row[2]))
    y = np.array(json.loads(row[3]))
    X_grid = np.array(json.loads(row[4]))
    kde_class0 = np.array(json.loads(row[5]))
    mean_class0 = np.array(json.loads(row[6]))

    return {
        'uncertainty': uncertainty,
        'vmax': vmax,
        'X': X,
        'y': y,
        'X_grid': X_grid,
        'kde_class0': kde_class0,
        'mean_class0': mean_class0
    }

# Initialize the database when the script is run
if __name__ == "__main__":
    initialize_database()