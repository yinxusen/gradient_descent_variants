gradient_descent_variants
=========================

Serveral different variants of gradient descent in Spark are implemented here, in order to enhance both the statistical efficiency and hardware efficiency of GLM(Generalized Linear Model) in Spark.

**Affected algorithms in Spark**

* Logistic Regression

* SVM

* Linear Regression

* Lasso

* Ridge Regression

**Note:**

Lasso with the same code (two copies) has different performance on accuracy, it is strange. The one using Spark `GradientDescent` directly has very bad accuracy performance, while the one using `GradientDescentAnother`, which is copied from `GradientDescent` performances good!

I'll take lots of time on it later.
