import matplotlib.pyplot as plt
import numpy as np

val_error = np.loadtxt("../res/validation_error.csv", delimiter=",")
train_error = np.loadtxt("../res/train_loss.csv", delimiter=",")
plt.figure(figsize=(10, 5))
# plt.ylim((0,80))
plt.plot(*zip(*val_error))
plt.plot(*zip(*train_error))
plt.ylabel("Error")
plt.xlabel("Epoch")
plt.legend(["Validation Error","Training Error"])
plt.show()