# DecisionTree

Implementation of decision trees based on the minimum nescience principle.

Example of Classification:

```
from sklearn.datasets import load_breast_cancer
from NescienceDecisionTree import NescienceDecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = NescienceDecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

0.925531914893617
```

Example of Regression:

```
from sklearn.datasets import load_boston
from NescienceDecisionTree import NescienceDecisionTreeRegressor

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = NescienceDecisionTreeRegressor()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
0.6575005396334208
```

Check the accompanying Jupyter Notebook for more examples.

Check http://www.mathematicsunknown.com/ for more information about the Theory of Nescience.
