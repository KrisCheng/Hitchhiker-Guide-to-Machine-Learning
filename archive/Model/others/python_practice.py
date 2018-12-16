import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# numpy
# mylist = [[1,2,3], [3,4,5]]
# myarray = np.array(mylist)
# print(myarray)
# print(myarray.shape)
# print("First row: %s" % myarray[0])
# print("Last row: %d" % myarray[-1, -1])
# print("Specific row and col: %d" % myarray[0, 2])
# print("Whole col: %s" % myarray[:, 2])

# matplotlib
# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# plt.scatter(x, y)
# plt.xlabel('some x axis')
# plt.ylabel('some y axis')
# plt.show()

# pandas
# Series
# myarray = np.array([1, 2, 3])
# rownames = ['a', 'b', 'c']
# myseries = pd.Series(myarray, index = rownames)
# print(myseries['a'])
# print(myseries[1])

# DataFrame
myarray = np.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pd.DataFrame(myarray, index = rownames, columns = colnames)
print(mydataframe.one)
