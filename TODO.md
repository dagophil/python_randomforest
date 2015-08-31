# TODO

* Find a good value for alpha so the lasso can be replaced by a single coordinate descent
* Parallelize the grouped forest garrote.
* Compare the termination criteria and the other options using cross validation.
  
  
### Open questions

* Why is the algorithm with resample_count=200 not faster? Look at depth, number of nodes, prediction complexity.

