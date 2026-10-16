import numpy as np
import matplotlib.pyplot as plt
from utils import z_score_normalize
from linear_regression import LinearRegressionScratch

np.random.seed(42)
m=100
X = np.random.rand(m,2) * 10
true_w = np.array([3.0, 1.5])
true_b = 4.0

y = np.dot(X,true_w) + true_b   + np.random.rand(m) * 0.5

X_norm, mu, sigma = z_score_normalize(X)

model = LinearRegressionScratch(alpha = 0.05, n_iters = 400)
model.fit(X_norm, y)

print(f"Trained w: {model.w}")
print(f"Trained b; {model.b: .4f}")

iterations, cost = zip(*model.cost_history)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(iterations, cost,label="MSE Cost", color="blue", linewidth=2)
ax.set_title("Gradient Descent Learning curve from Scratch)", fontsize=14, pad=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Mean Squared ErrorCost", fontsize=12)
ax.grid(True, linestyle="--", alpha = 0.6)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("plots/linear_cost_curve.png", dpi=300)
plt.show()
print("Saved convergence plot to plots/linear_cost_curve.png")