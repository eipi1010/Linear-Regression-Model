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


Here are hints (not fixes) about what’s going wrong or risky in your implementation, ordered from most important to subtle:

1. Your stopping condition (step_size) is not meaningful

step_size is computed as

(np.sum(w_steps) + b_step) / (w_steps.shape[0] + 1)


This can cancel out (positive and negative gradients), becoming small even when updates are large.

Gradient descent usually stops based on:

gradient norm

change in loss

fixed number of iterations

👉 Hint: ask yourself “does this number really represent how big my update was?”

2. You are missing normalization by number of samples

Your gradients use:

2 * (X.T @ y_diff)


For MSE, gradients are usually averaged over n samples.

Without this, gradient magnitude depends on dataset size → unstable learning rates.

👉 Hint: how does dataset size affect your gradient magnitude?

3. Learning rate + gradient scale mismatch

You set lr = 1e-4, but your gradients may be very large.

This can cause:

divergence

oscillation

extremely slow convergence

👉 Hint: print np.linalg.norm(w_steps) to see what scale you’re updating at.

4. mean_square_error() recomputes predictions every time

Inside step() you call predict()

Inside get_weights() you also call mean_square_error() → another predict()

This is inefficient and can hide logic errors.

👉 Hint: cache predictions per iteration.

5. Bias gradient is inconsistent with weight gradient

You compute:

b_step = -lr * 2 * np.sum(y_diff)


But weights use matrix multiplication.

Both should conceptually follow the same averaging logic.

👉 Hint: check whether bias should also be averaged over samples.

6. Your reshape logic is fragile
np.asarray(X_train).reshape(len(X_train), len(X_train.axes[1]))


This assumes:

no missing columns

no unexpected reshaping

np.asarray(X_train) already has the correct shape.

👉 Hint: unnecessary reshaping can silently corrupt data.

7. No feature scaling

If features are on different scales, gradient descent will struggle.

This often looks like:

loss not decreasing

needing extremely small lr

👉 Hint: check X_train.std(axis=0).

8. No max-iteration safeguard

If step_size never drops below 0.0001, you loop forever.

👉 Hint: always cap iterations.

9. predict() assumes NumPy array input

If you pass a pandas DataFrame with mismatched column order → silent bugs.

👉 Hint: consistency of feature order matters.

Mental checklist to debug

Ask yourself:

Is my gradient mathematically correct?

Does my stopping rule reflect convergence?

Are my updates scaled properly?

Can the loop run forever?

If you want, next I can:

point out one exact line that causes divergence

compare this to the closed-form normal equation

or show how to diagnose gradient descent numerically without changing code