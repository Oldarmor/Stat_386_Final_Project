import pandas as pd
from importlib.resources import files


def get_data_path(filename: str):
    """Return a path-like object for a data file inside the package."""
    return files(__name__).joinpath(f"data/{filename}")

def read_data(file_path):
    """Reads data from a given file path as a csv."""
    sales = pd.read_csv(file_path)
    sales['Year'] = sales['Year'].astype('Int64')
    # Some CSVs may not include an index column named 'Unnamed: 0'. Drop it only if present.
    if 'Unnamed: 0' in sales.columns:
        sales.drop(columns=['Unnamed: 0'], inplace=True)
    return sales