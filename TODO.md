# TODO

* Find a good value for alpha so the lasso can be replaced by a single coordinate descent
* Parallelize the grouped forest garrote.
* Try the following in the prediction step:

  Current version: `predicted_class = argmax(mean(probabilities))`
  
  Todo version: `predicted_class = argmax(sum(label_count))`
  
* Compare the termination criteria.
  
  
### Open questions

* Why is the algorithm with resample_count=200 not faster? Look at depth, number of nodes, prediction complexity.

