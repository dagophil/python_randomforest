# TODO

* Try the following in the prediction step:

  Current version: `predicted_class = argmax(mean(probabilities))`
  
  Todo version: `predicted_class = argmax(sum(label_count))`
  
* Add termination criterion: Stop if log(number of labels in node) <= threshold.
* Compare the termination criteria.
* Put the forest garrote weights into the forest and merge the graph nodes accordingly.
  
  
### Thesis questions

* Why is the algorithm with resample_count=200 not faster? Look at depth, number of nodes, prediction complexity.
