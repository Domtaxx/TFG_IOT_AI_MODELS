import pandas as pd

file_path = 'AI_models/file2.csv'
df = pd.read_csv(file_path)

new_column_name = 'target'
new_column_value = 'legitimate'
df[new_column_name] = new_column_value

df.to_csv(file_path, index=False)