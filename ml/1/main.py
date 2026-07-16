import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import seaborn as sns

# def read_and_fix_csv(filepath):
#     df = pd.read_csv(filepath)
#     if len(df.columns) == 1:
#         cols = df.columns[0].split(',')
#         df = df.iloc[:, 0].str.split(',', expand=True)
#         df.columns = cols
#         df = df.apply(pd.to_numeric)
#     return df.drop(columns=['id'], errors='ignore')
#
# train_data = read_and_fix_csv('ssz_train.csv')
# test_data = read_and_fix_csv('ssz_test.csv')
#
# X_train, y_train = train_data.drop(columns=['cardio']), train_data['cardio']
# X_test, y_test = test_data.drop(columns=['cardio']), test_data['cardio']
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train_scaled, y_train)



sns.set_theme()
rng = np.random.RandomState(0)

X1 = rng.randn(50, 2) + np.array([4,4])
X2 = rng.randn(50,2) + np.array([-4,4])
X3 = rng.randn(50, 2) + np.array([4,-4])
X4 = rng.randn(50, 2) + np.array([-4,-4])
X = np.concatenate([X1, X2, X3, X4])
y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)

plt.figure(figsize=(15,10))
plt.scatter(X[:,0], X[:,1], s=30, c=y, cmap=plt.cm.coolwarm)

show()