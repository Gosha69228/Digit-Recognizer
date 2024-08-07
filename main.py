import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# def preprocess_data(input_file):
#     df = pd.read_csv(input_file)
#
#     # first_column = df.iloc[:, 0]
#     # #  числовые столбцы, кроме первого
#     # numerical_cols = df.columns[1:]
#     # df_numerical = df[numerical_cols]
#     # # Применение Min-Max нормализацию к числовым столбцам
#     # scaler = MinMaxScaler()
#     # df_normalized = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_cols)
#     # df_result = pd.concat([first_column, df_normalized], axis=1)


input_file = "output_normalized.csv"
df = pd.read_csv(input_file)

if 'Id' not in df.columns:
    df['Id'] = range(len(df))

model = RandomForestClassifier(n_estimators=100, random_state=42)

known_Price = df[df['label'].notnull()]
unknown_Price = df[df['label'].isnull()].drop(columns=['label'])

X_train_Price = known_Price.drop(columns=['label'])
y_train_Price = known_Price['label']

model.fit(X_train_Price, y_train_Price)

predicted_Price = model.predict(unknown_Price)

df.loc[df['label'].isnull(), 'label'] = predicted_Price
df = df[['Id', 'label']]
df.label = df.label.astype(int)
df.iloc[42000:].to_csv('kag.csv', index=False)