# pyrefly: ignore [missing-import]
from src.data_loader import load_data

df = load_data()

print(df.head())