# TODO

* In prediction step:

  Current version: `predicted_class = argmax(mean(probabilities))`
  Todo version: `predicted_class = argmax(sum(label_count))`
* In each node: Do not save the total label count, but the difference to the parent node (needed for forest garrote).
* Implement forest garrote. Maybe use lasso instead of lars, since sklearn lasso already has the positive-constraint (`min gamma >= 0` in equation (13) in the forest garrote paper).

  `alphas, coefs, dual_gaps = linear_model.lasso_path(X, y, verbose=True, positive=True)`
