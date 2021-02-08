import pandas as pd
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data = pd.read_csv('Book_1.csv')

print(data)

a = data[data.columns[0]].values #first col
c = data[data.columns[-1]].values #last col

xs = np.array(a, dtype=np.float64)  #add to array and convert into float
ys = np.array(c, dtype=np.float64)  

print(xs)
print(ys)

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))    #slope of lone
    b = mean(ys) - m * mean(xs)              #intercept of line
    return m, b


m, b = best_fit_slope_and_intercept(xs, ys)

print("y = "+np.str_(m)+"x + "+np.str_(b))  #line equation
# regression_line = [(m*x)+b for x in xs]
regression_line = []
for x in xs:
    regression_line.append((m*x)+b)   

plt.scatter(xs,ys,color='g')
plt.plot(xs, regression_line)
plt.show()