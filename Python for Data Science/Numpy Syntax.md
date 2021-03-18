## Creating an array


```python
import numpy as np
```


```python
list1 = [1,2,3,4,5]

array_from_list = np.array(list1)  # or np.array([1,2,3,4,5]). This is 1D array
array_from_list
```




    array([1, 2, 3, 4, 5])




```python
array_from_tuple = np.array((1,2,3,4,5))
array_from_tuple
```




    array([1, 2, 3, 4, 5])



#### Numpy arrays are homogenous. That is all the elements in the array have to be of same datatype


```python
array_2d = np.array([[1,2,3], [4,5,6]])  #Pass a list of lists if you have to create 2D Arrays
array_2d
```




    array([[1, 2, 3],
           [4, 5, 6]])



### Other methods of creating an array are as follows:

#### np.arange(arg1, arg2, step) - To create an evenly spaced array. Note that arg2 is excluded from the result.


```python
np.arange(1 , 20, 2)
```




    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])




```python
np.arange(20)  #if just one arg is passed it takes that arg as 'arg2' and step by default as 1. Also note that it starts from 0
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])



#### np.zeros(arg) - To create an array of 'arg' number of zeroes


```python
np.zeros(5)   # By default it will return float values.
```




    array([0., 0., 0., 0., 0.])




```python
np.zeros(5, dtype= int)   # we can change dtype by passing arg as 'dtype'
```




    array([0, 0, 0, 0, 0])




```python
np.ones(5, dtype= int)   # similarly ones
```




    array([1, 1, 1, 1, 1])




```python
np.ones((5,4))    # Similarly we can creat 2D arrays for zeros.
```




    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])



#### np.linspace(arg1, arg2, arg3) - To create an array of specified evenly spaced numbers between given intervals. Please note the 'arg1' and 'arg2' both are included. 'arg3' is number of elements you want in the result.


```python
np.linspace(1, 20, 10)
```




    array([ 1.        ,  3.11111111,  5.22222222,  7.33333333,  9.44444444,
           11.55555556, 13.66666667, 15.77777778, 17.88888889, 20.        ])



#### np.eye(arg) - To create (arg*arg) identity matrix


```python
np.eye(2)
```




    array([[1., 0.],
           [0., 1.]])



#### To create an array of random variables:
1 - From Uniform Distribution i.e between 0 and 1 \
2 - From standard Normal Distribution \
3 - Random integers in given interval


```python
np.random.rand(5)  # 'arg' is the number of values needed from UNIFORM DISTRIBUTION
```




    array([0.49677541, 0.90801866, 0.24105841, 0.31583176, 0.70368364])




```python
np.random.rand(5,4)   # To get 2D array 
```




    array([[0.40060315, 0.69951369, 0.76731604, 0.99228396],
           [0.7105502 , 0.11386213, 0.50807106, 0.02357548],
           [0.64256844, 0.82927103, 0.67255387, 0.54773636],
           [0.62309893, 0.16661983, 0.23692307, 0.54939187],
           [0.23615707, 0.37658835, 0.7509972 , 0.68237739]])




```python
np.random.randn(5)  #'arg' is number of values needed from STANDARD NORMAL DISTRIBUTION
```




    array([ 0.34920857,  0.96214072, -0.09344336, -0.33045297,  0.61370805])




```python
np.random.randn(5,4)
```




    array([[ 0.57274397, -0.3662771 ,  0.43532685,  0.31563745],
           [-0.36493581, -0.12328378,  2.06373906, -1.18583175],
           [-1.94399592, -0.19296823,  0.13993823, -0.56171989],
           [-0.12216412, -0.63250391,  1.20629521, -0.27698503],
           [-0.30765962, -0.00966756, -0.69163603, -0.32919575]])




```python
np.random.randint(0,20,5)   # To get 5 random integers from 0 to 19(arg2 exculded). If arg3 not specified then 1 value is returned
```




    array([ 1, 18,  3, 18,  6])




```python
np.random.random((3,3))   # Pass a tuple if you want multi dimensional array
```




    array([[0.55462432, 0.35920371, 0.67345534],
           [0.82870826, 0.21924166, 0.01798282],
           [0.49697061, 0.38994154, 0.41810924]])



#### np.full(arg1 , arg2) - To create an array of 'arg2' int/float 'arg1' times


```python
np.full (5 , 10)
```




    array([10, 10, 10, 10, 10])




```python
np.full((3,4), 10)
```




    array([[10, 10, 10, 10],
           [10, 10, 10, 10],
           [10, 10, 10, 10]])



#### np.tile( array, arg2)  - This function is used to create a new array by repeating an existing array for 'arg2' number of times


```python
np.tile(np.array([1,2,3,4]), 2)
```




    array([1, 2, 3, 4, 1, 2, 3, 4])




```python
np.tile(np.array([1,2,3,4]), (2,2))  # Note:array multipled by 2 in row and in column also
```




    array([[1, 2, 3, 4, 1, 2, 3, 4],
           [1, 2, 3, 4, 1, 2, 3, 4]])



### Operations that we can do on Numpy Arrays:

#### Calculate element-wise product of 2 arrays:


```python
arr_1 = np.array([1,2,3,4])   #For lists, we will have to use lambda with map function
arr_2 = np.array([5,6,7,8])
arr_1*arr_2
```




    array([ 5, 12, 21, 32])



#### Rounding off the values in the array:

np.round(array_name, decimals)


```python
random=np.random.rand(5,4)
random
```




    array([[0.89257429, 0.7485368 , 0.71142753, 0.79757671],
           [0.62494539, 0.92301408, 0.1958532 , 0.37862384],
           [0.19746691, 0.11181457, 0.5906272 , 0.16165135],
           [0.9805932 , 0.71722276, 0.9996579 , 0.50816858],
           [0.98857773, 0.33417449, 0.45141936, 0.97898859]])




```python
np.round(random,2)
```




    array([[0.89, 0.75, 0.71, 0.8 ],
           [0.62, 0.92, 0.2 , 0.38],
           [0.2 , 0.11, 0.59, 0.16],
           [0.98, 0.72, 1.  , 0.51],
           [0.99, 0.33, 0.45, 0.98]])



#### Squared array:


```python
arr_1**2   # For lists we can do by list comprehension
```




    array([ 1,  4,  9, 16], dtype=int32)



#### Reshaping Arrays : array_name.reshape(rows,cols)


```python
np.arange(20).reshape(4,5)
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])



#### Stacking and splitting arrays :

1 - np.hstack() - For horizontal stacking number of rows must be same

2 - np.vstack() - For vertical stacking number of cols must be same

Alternative syntax : np.stack((arr1,arr2) , axis=1/0)


```python
arr_1
```




    array([1, 2, 3, 4])




```python
arr_2
```




    array([5, 6, 7, 8])




```python
np.hstack((arr_1,arr_2))    # Pass arrays as a tuple or it will throw an error
```




    array([1, 2, 3, 4, 5, 6, 7, 8])




```python
np.vstack((arr_1,arr_2))
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])



#### Return indices of non-zero elements


```python
a=np.array([1,3,0,0,5])
nz=np.nonzero(a)
nz                             # Returns array of indices
```




    (array([0, 1, 4], dtype=int64),)



#### Extract a diagonal of matrix


```python
np.diag(np.arange(4),k=0)  #  It will put values of array in the diagonal
```




    array([[0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 3]])




```python
d=np.diag(np.arange(4),k=-1) 
d#If k<0, the diagonal below original diagonal will take values of array
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 3, 0]])



#### Changing dtype of array


```python
d.dtype
```




    dtype('int32')




```python
f=d.astype(float)
f.dtype
f
```




    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 2., 0., 0.],
           [0., 0., 0., 3., 0.]])



### Structure and content of arrays


```python

```

#### array_name.shape : It is an attribute to determine number of rows and columns(rows,columns)


```python
array_2d.shape
```




    (2, 3)



#### array_name.dtype : It determines the datatype


```python
array_2d.dtype
```




    dtype('int32')




```python
heter=np.array([1,2,3,'4'])
heter.dtype
```




    dtype('<U11')



#### array_name.ndim : It gives the dimension or the axes of the array


```python
array_from_list.ndim
```




    1




```python
array_2d.ndim
```




    2



#### array_name.itemsize : It determines the memory used by each element of an array in bytes.


```python
array_from_list.itemsize
```




    4




```python
array_2d.itemsize
```




    4




```python
heter.itemsize
```




    44



#### Transpose of a multidimensional array


```python
array_2d.T
```




    array([[1, 4],
           [2, 5],
           [3, 6]])



### Slicing and Dicing through Arrays


```python
array_from_list
```




    array([1, 2, 3, 4, 5])




```python
array_from_list[2]  # Same as in lists
```




    3




```python
array_from_list[[1,2,3]]    # To fetch multiple elements pass list of indices as an arg
```




    array([2, 3, 4])




```python
array_from_list[2:]        # 3rd element onwards
```




    array([3, 4, 5])




```python
array_from_list[:2]      #Upto 2nd element
```




    array([1, 2])




```python
array_from_list[2:4]
```




    array([3, 4])




```python
array_2d
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
array_2d[1,2]
```




    6




```python
array_2d[1, :]           # to fetch 2nd row and all columns
```




    array([4, 5, 6])




```python
array_2d[:,1]           #Similarly fetching all rows and 2nd column
```




    array([2, 5])




```python
array_2d[:, 0:2]        # all rows and columns in range o to 2
```




    array([[1, 2],
           [4, 5]])




```python
array_2d[: , (0,2)]     # all rows and 1st and 3rd column..If you want separate rows/cols, pass indices in a tuple
```




    array([[1, 3],
           [4, 6]])




```python
for row in array_2d:        # We can also iterate using for loop, but numpy arrays are not meant to be iterated using for loops
    print(row)
```

    [1 2 3]
    [4 5 6]
    

### Basic Mathematical operations on Arrays


```python
arr_1
```




    array([1, 2, 3, 4])




```python
arr_1*2
```




    array([2, 4, 6, 8])




```python
arr_1/2
```




    array([0.5, 1. , 1.5, 2. ])




```python
arr_1+2
```




    array([3, 4, 5, 6])




```python
arr_1-2
```




    array([-1,  0,  1,  2])




```python
arr_1=arr_1**2
arr_1
```




    array([ 1,  4,  9, 16], dtype=int32)




```python
np.sqrt(arr_1)
```




    array([1., 2., 3., 4.])




```python
np.exp(arr_1)
```




    array([2.71828183e+00, 5.45981500e+01, 8.10308393e+03, 8.88611052e+06])




```python
np.sin(arr_1)
```




    array([ 0.84147098, -0.7568025 ,  0.41211849, -0.28790332])




```python
np.cos(arr_1)
```




    array([ 0.54030231, -0.65364362, -0.91113026, -0.95765948])




```python
np.arcsin(arr_1)      #Opposite of sine
```

    C:\Users\Pratik\anaconda3\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in arcsin
      """Entry point for launching an IPython kernel.
    




    array([1.57079633,        nan,        nan,        nan])




```python
np.degrees(1)          # to convert radians in degrees
```




    57.29577951308232




```python
np.max(arr_1)          # gives max value in array
```




    16




```python
np.argmax(arr_1)        #Gives index position of max value. Note: this method doesn't work on multidimensional arrays
```




    3




```python
np.min(arr_1)
```




    1




```python
np.argmin(arr_1)      # Note: this method doesn't work on multidimensional arrays
```




    0




```python
np.square(arr_1)
```




    array([  1,  16,  81, 256], dtype=int32)




```python
np.max(array_2d)
```




    6




```python
np.log(arr_1)
```




    array([0.        , 1.38629436, 2.19722458, 2.77258872])



### User-defined functions on arrays

If you want to apply a specific function on numpy array, you can use np.vectorize() method

np.vectorize(function)


```python
f = np.vectorize(lambda x: x/(x+1))   # 'f' 's datatype is an object.
f(arr_1)
```




    array([0.5       , 0.8       , 0.9       , 0.94117647])




```python
f(array_2d)               # You can apply this vectorised function multiple times ahead.
```




    array([[0.5       , 0.66666667, 0.75      ],
           [0.8       , 0.83333333, 0.85714286]])



### Basic Linear Algebra Operations


```python
array_inv = np.array([[1,2,3],[4,5,6],[7,8,9]])
array_inv
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
np.linalg.inv(array_inv)               # Inverse of matrix
```




    array([[ 3.15251974e+15, -6.30503948e+15,  3.15251974e+15],
           [-6.30503948e+15,  1.26100790e+16, -6.30503948e+15],
           [ 3.15251974e+15, -6.30503948e+15,  3.15251974e+15]])




```python
np.linalg.det(array_inv)             # Determinant of matrix
```




    -9.51619735392994e-16




```python
np.linalg.eig(array_inv)         # Eigen values and eigen vectors of matrix
```




    (array([ 1.61168440e+01, -1.11684397e+00, -9.75918483e-16]),
     array([[-0.23197069, -0.78583024,  0.40824829],
            [-0.52532209, -0.08675134, -0.81649658],
            [-0.8186735 ,  0.61232756,  0.40824829]]))




```python
np.dot(array_inv[(0,1),(0,1)],array_inv[(1,2),(1,2)])      # dot product of a matrix
```




    50



### Broadcasting
when we subset an array using '[:]', we just see the sub-view of that array and note that data is not copied in the subset


```python
arr_1[0:2]
```




    array([1, 4], dtype=int32)




```python
arr_1[0:2] = 100      #Broadcasting. While broadcasting, original array gets updated
arr_1
```




    array([100, 100,   9,  16], dtype=int32)



### Saving and Loading an Array

1 - np.save('file_name.npy', array_name) : This stores a single array.

2 - np.load('filename.npy') : This loads the saved array

3 - np.savez('filename.npz', 'a'=arr1, 'b'=arr2) : This saves the 2 arrays in a zip file.

4 - abc = np.load('filename.npz') 
    abc['a'] : loads arr1
    abc['b'] : loads arr2

5 - np.savetxt('filename.txt', arrname, delimiter=',')

5 - np.loadtxt ('filename.txt', delimiter =',')

### np.where() in detail

Syntax : np.where (condition, x, y)  : If condition is true, it returns 'x' else returns 'y'. It return a new matrix


```python
matrix = np.linspace(0,20,9).reshape(3,3)
matrix
```




    array([[ 0. ,  2.5,  5. ],
           [ 7.5, 10. , 12.5],
           [15. , 17.5, 20. ]])




```python
matrix2=np.where (matrix<10 , 1, 100)
matrix2
```




    array([[  1,   1,   1],
           [  1, 100, 100],
           [100, 100, 100]])




```python
np.where (matrix==1,100,matrix)
```




    array([[ 0. ,  2.5,  5. ],
           [ 7.5, 10. , 12.5],
           [15. , 17.5, 20. ]])




```python
np.where((matrix2==1) & (matrix2==100), -1,10)  #Multple filters for a single element
```




    array([[10, 10, 10],
           [10, 10, 10],
           [10, 10, 10]])




```python
np.where(matrix2==1,matrix2*100,matrix2)     # To process elements that satisy condition
```




    array([[100, 100, 100],
           [100, 100, 100],
           [100, 100, 100]])



### Print Numpy Version and config


```python
np.__version__
```




    '1.18.1'




```python
np.show_config()
```

    blas_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Pratik/anaconda3\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Pratik/anaconda3\\Library\\include']
    blas_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Pratik/anaconda3\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Pratik/anaconda3\\Library\\include']
    lapack_mkl_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Pratik/anaconda3\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Pratik/anaconda3\\Library\\include']
    lapack_opt_info:
        libraries = ['mkl_rt']
        library_dirs = ['C:/Users/Pratik/anaconda3\\Library\\lib']
        define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
        include_dirs = ['C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\include', 'C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.0.117\\windows\\mkl\\lib', 'C:/Users/Pratik/anaconda3\\Library\\include']
    


```python

```
