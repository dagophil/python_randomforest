# TODO

* In prediction step:

  Current version: `predicted_class = argmax(mean(probabilities))`
  
  Todo version: `predicted_class = argmax(sum(label_count))`
* In each node: Do not save the total label count, but the difference to the parent node (needed for forest garrote).
* Implement forest garrote. Maybe use lasso instead of lars, since sklearn lasso already has the positive-constraint (`min gamma >= 0` in equation (13) in the forest garrote paper). Or maybe use logistic regression with L1 regularization (is the result sparse?).

  `alphas, coefs, dual_gaps = sklearn.linear_model.lasso_path(X, y, verbose=True, positive=True)`

* Add termination criterion: Stop if log(number of labels in node) <= threshold.
* Compare the termination criteria.
  
  
### Thesis questions

* Why is the algorithm with resample_count=200 not faster? Look at depth, number of nodes, prediction complexity.

