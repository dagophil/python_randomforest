## Forest Garrote training:
* The sklearn Lars algorithm does not accept sparse input, so it cannot be used.
* If you compute the SVM weights instead of the Lasso weights, the resulting node weights are far from sparse.
* Precomputing the gram matrix for large trees is (a) computationally expensive and (b) memory consuming, since the result is of shape (num_nodes, num_nodes) and probably not sparse. Therefore the trees are gathered in small groups (4-5 trees) and the forest garrote is computed on each group.

