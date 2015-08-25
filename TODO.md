# TODO

* Find a good value for alpha so the lasso can be replaced by a single coordinate descent
* After the grouped forest garrote: Improve the weights using SVM or something similar.
* Parallelize the grouped forest garrote.
* Try the following in the prediction step:

  Current version: `predicted_class = argmax(mean(probabilities))`
  
  Todo version: `predicted_class = argmax(sum(label_count))`
  
* Add termination criterion: Stop if log(number of labels in node) <= threshold.
* Add termination criterion (Köthe white paper): Stop if lga(n_left + 1) + lga(n_right + 1) - lga(n_left + n_right + 1) >= log(a / 2) for some small a and lga = loggamma.
* Compare the termination criteria.
* Compare matrix rank of the leaf index vectors before and after forest garrote.
  
  
### Thesis questions

* Why is the algorithm with resample_count=200 not faster? Look at depth, number of nodes, prediction complexity.

