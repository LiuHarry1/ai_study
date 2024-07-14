import numpy as np

#matmul() 用于计算两个数组的矩阵乘积。示例如下
def matmul_test():
    array1 = np.array([
        [[1.0, 3], [1, 1], [2, 3]]
    ])
    array2 = np.array([[2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                       [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0], ])
    result = np.matmul(array1, array2)
    print(result)
    ''' 
    [[[5. 4. 1. 3. 3. 0. 0. 4. 4. 0. 1. 0.]
      [3. 2. 1. 1. 1. 0. 0. 2. 2. 0. 1. 0.]
      [7. 5. 2. 3. 3. 0. 0. 5. 5. 0. 2. 0.]]]
    '''

def multiple_test():
    array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ndmin=3)
    array2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], ndmin=3)
    result = np.multiply(array1, array2)
    print(result)
    '''
     [[[ 9 16 21]
      [24 25 24]
      [21 16  9]]]
    '''

#dot() 函数用于计算两个矩阵的点积。如下所示：
#https://blog.csdn.net/KUNPLAYBOY/article/details/121038134
def dot_test():
    array1=np.array([[1,2,1],[0,3,2],[1,1,2]])
    array2=np.array([[2,1,1],[1,0,2],[2,1,1]])
    result=np.dot(array1,array2)
    print(result)

    array1 = np.array([[1, 2, 1], [1, 3, 2], [1, 1, 2]])
    array2 = np.array([2, 1, 1])
    result = np.dot(array1, array2)
    print(result)


def test1():
    # Sample 2D array
    data = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])
    print(data.shape)
    # Calculate the mean along axis 0 (mean of each column)
    mean_values_axis0 = np.mean(data, axis=0)
    # Calculate the mean along axis 1 (mean of each row)
    mean_values_axis1 = np.mean(data, axis=1)
    mean_values_axis01 = np.mean(data, axis=(0, 1), keepdims=True)
    print("Mean along axis 0:", mean_values_axis0)
    print("Mean along axis 1:", mean_values_axis1)
    print("Mean along axis 1:", mean_values_axis01)


if __name__ == '__main__':
    # test1()
    multiple_test()
    # dot_test()
    # matmul_test()
