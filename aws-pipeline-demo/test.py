import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('./cities-job-satisfaction.csv')
print(df.head())
x = df['last_year']
y = df['this_year']
m, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, m*x + b, color='red')
plt.savefig('books_read.jpg')
