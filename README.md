#LINEAR REGRESSION MODEL

1. Define the model
Output assumed is of form y=wx + wx + wx + b
Each w is a weight
b is the intercept
y is the predicted value

2. Choose a loss function
Use MSE

3. Initialise Parameters

4. Measure predition error
For each data point:
Use current weights to predict y
Compare it with the true y
Compute the loss

5. Compute gradients
Calculate the partial derivative of the loss with respect to each parameter

6. Update parameters (optimisation)
Adjust all weights and the bias slightly in the direction that reduces the loss

7. Iterate until convergence
Keep looping until the loss stops decreasingly significantly or
a fixed number of iterations is reached

8. Use the trained model
Once trained:
Plug in the new feature values
Compute the weighted sum
Output the prediction