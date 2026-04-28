import numpy as np
import pandas as pd
from scipy import stats

data = [72, 15, 18, 22, 25, 35, 28, 30, 35, 40]
mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=True).mode[0]
range_val = np.max(data) - np.min(data)
variance_val = np.var(data)
std_dev = np.std(data)

print(f"Data: {data}")
print(f"Mean: {mean_val}, Median: {median_val}, Mode: {mode_val}")
print(f"Range: {range_val}, Variance: {variance_val:.2f}, Std Dev: {std_dev:.2f}")