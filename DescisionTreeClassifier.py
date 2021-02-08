import numpy as np
import pandas as pd
import pydotplus
from sklearn.externals.six import StringIO  
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


# reading Dataset
dataset = pd.read_csv('CosmeticShop.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

# Perform Label encoding
print(X)
print(y)

le = LabelEncoder()

X = X.apply(le.fit_transform)
print("X")
print(X)


regressor = DecisionTreeClassifier()
regressor.fit(X.iloc[:, 0:4], y)

# Predict value for the given Expression
X_in = np.array([1, 1, 0, 0])   #8 th number row 
y_pred = regressor.predict([X_in])
print("Prediction:", y_pred)

dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('tree1.png')
