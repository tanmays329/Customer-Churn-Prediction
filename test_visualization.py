# pyrefly: ignore [missing-import]
from src.data_loader import load_data
# pyrefly: ignore [missing-import]
from src.visualization import create_churn_pie

df = load_data()

fig = create_churn_pie(df)

fig.show()