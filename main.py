import pandas as pd

# Import and analyze data
df = pd.read_csv('./data/SBAnational.csv')
df_copy = df.copy()

# Save our model and use it for other SBA apps
#xgboost.save_model("./sba_default.xgb")
