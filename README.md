# A python/cython random forest implementation
This repository contains python/cython code with a random forest implementation.

### Installation:
```
python setup.py build_ext --inplace
```

### Example usage in your python code:
```
rf = randomforest.RandomForestClassifier(n_estimators=100)
rf.fit(train_data, train_labels)
pred_labels = rf.predict(test_data)
```
