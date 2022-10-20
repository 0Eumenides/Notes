#  Numpy

## 创建数组

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```
参数说明：

| 名称   | 描述                                                      |
| :----- | :-------------------------------------------------------- |
| object | 数组或嵌套的数列                                          |
| dtype  | 数组元素的数据类型，可选                                  |
| copy   | 对象是否需要复制，可选                                    |
| order  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok  | 默认返回一个与基类类型一致的数组                          |
| ndmin  | 指定生成数组的最小维度                                    |

```python
# 最小维度  
import numpy as np 
a = np.array([1, 2, 3, 4, 5], ndmin =  2)  
print (a)
```

**从数值范围创建数组**

```python
numpy.arange(start, stop, step, dtype)
```

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

参数说明：

| 参数       | 描述                                                         |
| :--------- | :----------------------------------------------------------- |
| `start`    | 序列的起始值                                                 |
| `stop`     | 序列的终止值，如果`endpoint`为`true`，该值包含于数列中       |
| `num`      | 要生成的等步长的样本数量，默认为`50`                         |
| `endpoint` | 该值为 `true` 时，数列中包含`stop`值，反之不包含，默认是True。 |
| `retstep`  | 如果为 True 时，生成的数组中会显示间距，反之不显示。         |
| `dtype`    | `ndarray` 的数据类型                                         |

```python
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```

| 参数       | 描述                                                         |
| :--------- | :----------------------------------------------------------- |
| `start`    | 序列的起始值为：base ** start                                |
| `stop`     | 序列的终止值为：base ** stop。如果`endpoint`为`true`，该值包含于数列中 |
| `num`      | 要生成的等步长的样本数量，默认为`50`                         |
| `endpoint` | 该值为 `true` 时，数列中中包含`stop`值，反之不包含，默认是True。 |
| `base`     | 对数 log 的底数。                                            |
| `dtype`    | `ndarray` 的数据类型                                         |

## 数据类型

| 名称       | 描述                                                         |
| :--------- | :----------------------------------------------------------- |
| bool_      | 布尔型数据类型（True 或者 False）                            |
| int_       | 默认的整数类型（类似于 C 语言中的 long，int32 或 int64）     |
| intc       | 与 C 的 int 类型一样，一般是 int32 或 int 64                 |
| intp       | 用于索引的整数类型（类似于 C 的 ssize_t，一般情况下仍然是 int32 或 int64） |
| int8       | 字节（-128 to 127）                                          |
| int16      | 整数（-32768 to 32767）                                      |
| int32      | 整数（-2147483648 to [2147483647](tel:2147483647)）          |
| int64      | 整数（-9223372036854775808 to 9223372036854775807）          |
| uint8      | 无符号整数（0 to 255）                                       |
| uint16     | 无符号整数（0 to 65535）                                     |
| uint32     | 无符号整数（0 to [4294967295](tel:4294967295)）              |
| uint64     | 无符号整数（0 to 18446744073709551615）                      |
| float_     | float64 类型的简写                                           |
| float16    | 半精度浮点数，包括：1 个符号位，5 个指数位，10 个尾数位      |
| float32    | 单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位      |
| float64    | 双精度浮点数，包括：1 个符号位，11 个指数位，52 个尾数位     |
| complex_   | complex128 类型的简写，即 128 位复数                         |
| complex64  | 复数，表示双 32 位浮点数（实数部分和虚数部分）               |
| complex128 | 复数，表示双 64 位浮点数（实数部分和虚数部分）               |

## 数组属性

**数组维度**

```python
a = np.array([[1,2,3],[4,5,6]])  
print (a.shape)

# 调整维度
a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2)  
```

**元素大小**

```python
# 数组的 dtype 为 int8（一个字节）  
x = np.array([1,2,3,4,5], dtype = np.int8)  
print (x.itemsize)
# 输出1

# 数组的 dtype 现在为 float64（八个字节） 
y = np.array([1,2,3,4,5], dtype = np.float64)  
print (y.itemsize)
# 输出8
```

## 切片和索引

```python
import numpy as np
 
a = np.arange(10)  
b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
print(b)
# 输出[2  4  6]
```

对多维数组

```python
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])

# 输出
# 从数组索引 a[1:] 处开始切割
# [[3 4 5]
#  [4 5 6]]
```

切片还可以包括省略号 **…**，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。

```python
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素
# 输出
# [2 4 5]

# [3 4 5]

# [[2 3]
#  [4 5]
#  [5 6]]
```

## 高级索引

```python
# 获取数组中 (0,0)，(1,1) 和 (2,0) 位置处的元素。
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]  
print (y)
```

获取四个角元素

```python
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)

# 输出
# 这个数组的四个角元素是：
# [[ 0  2]
#  [ 9 11]]
```

**布尔索引**

```python
import numpy as np

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
# 现在我们会打印出大于 5 的元素
print('大于 5 的元素是：')
print(x > 5)
print('\n')
print(x[x > 5])

# 输出
# 大于 5 的元素是：
# [[False False False]
#  [False False False]
#  [ True  True  True]
#  [ True  True  True]]

# [ 6  7  8  9 10 11]
```

使用` ~`（取补运算符）来过滤 NaN。

```python
a = np.array([np.nan,  1,2,np.nan,3,4,5])  
print (a[~np.isnan(a)])

# 输出
# [ 1.   2.   3.   4.   5.]
```

**花式索引**

一维数组只有一个轴 **axis = 0**，所以一维数组就在 **axis = 0** 这个轴上取值：

```python
x = np.arange(9)
x2 = x[[0, 6]] # 使用花式索引
print(x2)

# 输出
# [0 6]
```

对于二维数组

```python
x=np.arange(32).reshape((8,4))
# 输出表 4, 2, 1, 7 对应的行
print (x[[4,2,1,7]])

# 输出
# [[16 17 18 19]
#  [ 8  9 10 11]
#  [ 4  5  6  7]
#  [28 29 30 31]]
```

## 广播

- 如果数组a和b的形状相同，进行算数运算时结果就是两个数组对应位相乘。

- 形状不同时，numpy将触发广播机制

  ```python
  a = np.array([[ 0, 0, 0],
             [10,10,10],
             [20,20,20],
             [30,30,30]])
  b = np.array([0,1,2])
  print(a + b)
  
  # 输出
  # [[ 0  1  2]
  #  [10 11 12]
  #  [20 21 22]
  #  [30 31 32]]
  ```



## 数组操作

### 改变数组形状

| 函数      | 描述                                               |
| :-------- | :------------------------------------------------- |
| `reshape` | 不改变数据的条件下修改形状                         |
| `flat`    | 数组元素迭代器                                     |
| `flatten` | 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组 |
| `ravel`   | 返回展开数组                                       |

numpy.ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组。

numpy.ravel()修改会影响原始数组。

```python
a = np.arange(8).reshape(2,4)
print (a.flatten())
print (a.ravel())
 
# 输出
# [0 1 2 3 4 5 6 7]
# [0 1 2 3 4 5 6 7]
```

### 翻转数组

| 函数        | 描述                       |
| :---------- | :------------------------- |
| `transpose` | 对换数组的维度             |
| `ndarray.T` | 和 `self.transpose()` 相同 |
| `rollaxis`  | 向后滚动指定的轴           |
| `swapaxes`  | 对换数组的两个轴           |

```python
numpy.transpose(arr, axes)
```

参数说明:

- `arr`：要操作的数组
- `axes`：整数列表，对应维度，通常所有维度都会对换。

```python
a = np.arange(12).reshape(3,4)
print (np.transpose(a))

# 输出
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
```

numpy.ndarray.T 类似 numpy.transpose：

### 修改数组维度

| 维度           | 描述                       |
| :------------- | :------------------------- |
| `broadcast`    | 产生模仿广播的对象         |
| `broadcast_to` | 将数组广播到新形状         |
| `expand_dims`  | 扩展数组的形状             |
| `squeeze`      | 从数组的形状中删除一维条目 |

numpy.expand_dims 函数通过在指定位置插入新的轴来扩展数组形状，函数格式如下:

```python
 numpy.expand_dims(arr, axis)
```

参数说明：

- `arr`：输入数组
- `axis`：新轴插入的位置

```python

import numpy as np
 
x = np.array(([1,2],[3,4]))
 
print ('数组 x：')
print (x)
print ('\n')
y = np.expand_dims(x, axis = 0)
 
print ('数组 y：')
print (y)
print ('\n')
 
print ('数组 x 和 y 的形状：')
print (x.shape, y.shape)
print ('\n')
# 在位置 1 插入轴
y = np.expand_dims(x, axis = 1)
 
print ('在位置 1 插入轴之后的数组 y：')
print (y)
print ('\n')
 
print ('x.ndim 和 y.ndim：')
print (x.ndim,y.ndim)
print ('\n')
 
print ('x.shape 和 y.shape：')
print (x.shape, y.shape)
```

输出结果为：

```python
数组 x：
[[1 2]
 [3 4]]


数组 y：
[[[1 2]
  [3 4]]]


数组 x 和 y 的形状：
(2, 2) (1, 2, 2)


在位置 1 插入轴之后的数组 y：
[[[1 2]]

 [[3 4]]]


x.ndim 和 y.ndim：
2 3


x.shape 和 y.shape：
(2, 2) (2, 1, 2)
```

numpy.squeeze 函数从给定数组的形状中删除一维的条目，函数格式如下：

```python
numpy.squeeze(arr, axis)
```

参数说明：

- `arr`：输入数组
- `axis`：整数或整数元组，用于选择形状中一维条目的子集

```python
x = np.arange(9).reshape(1,3,3)
y = np.squeeze(x)
print (x.shape, y.shape)

# 输出
# (1, 3, 3) (3, 3)
```

### 连接数组

| 函数          | 描述                           |
| :------------ | :----------------------------- |
| `concatenate` | 连接沿现有轴的数组序列         |
| `stack`       | 沿着新的轴加入一系列数组。     |
| `hstack`      | 水平堆叠序列中的数组（列方向） |
| `vstack`      | 竖直堆叠序列中的数组（行方向） |

numpy.concatenate 函数用于沿指定轴连接相同形状的两个或多个数组，格式如下

```python
numpy.concatenate((a1, a2, ...), axis)
```

参数说明：

- `a1, a2, ...`：相同类型的数组
- `axis`：沿着它连接数组的轴，默认为 0

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print (np.concatenate((a,b)))
print (np.concatenate((a,b),axis = 1))

# 输出
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# [[1 2 5 6]
#  [3 4 7 8]]
```

numpy.stack 函数用于沿新轴连接数组序列，格式如下:

```python
numpy.stack(arrays, axis)
```

参数说明：

- `arrays`相同形状的数组序列
- `axis`：返回数组中的轴，输入数组沿着它来堆叠

```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print (np.stack((a,b),0))
print (np.stack((a,b),1))

# 输出
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]]

# [[[1 2]
#  [5 6]]

#  [[3 4]
#   [7 8]]]
```

Numpy.hstack和numpy.vstack是numpy.concatenate的变体

```python
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(np.hstack((a,b)))
print(np.concatenate((a,b),1))

# 输出
# [[1 2 5 6]
#  [3 4 7 8]]

# [[1 2 5 6]
#  [3 4 7 8]]
```

### 分割数组

| 函数     | 数组及操作                             |
| :------- | :------------------------------------- |
| `split`  | 将一个数组分割为多个子数组             |
| `hsplit` | 将一个数组水平分割为多个子数组（按列） |
| `vsplit` | 将一个数组垂直分割为多个子数组（按行） |

numpy.split 函数沿特定的轴将数组分割为子数组，格式如下：

```python
numpy.split(ary, indices_or_sections, axis)
```

参数说明：

- `ary`：被分割的数组
- `indices_or_sections`：如果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置（左开右闭） 
- `axis`：设置沿着哪个方向进行切分，默认为 0，横向切分，即水平方向。为 1 时，纵向切分，即竖直方向。

```python
a = np.arange(9)
print(np.split(a,3))
print(np.split(a,[4,7]))

# 输出
# [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
# [array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]
```

numpy.hsplit和numpy.vsplit则分别对应了numpy.split 的参数axis=0和axis=1的情况

### 数组元素的添加与删除

| 函数     | 元素及描述                                                   |
| :------- | :----------------------------------------------------------- |
| `resize` | 返回指定形状的新数组<br />numpy.resize(arr, shape)           |
| `append` | 将值添加到数组末尾<br />numpy.append(arr, values, axis=None) |
| `insert` | 沿指定轴将值插入到指定下标之前                               |
| `delete` | 删掉某个轴的子数组，并返回删除后的新数组                     |
| `unique` | 用于去除数组中的重复元素                                     |

```python
numpy.insert(arr, obj, values, axis)
```

参数说明：

- `arr`：输入数组
- `obj`：在其之前插入值的索引
- `values`：要插入的值
- `axis`：沿着它插入的轴，如果未提供，则输入数组会被展开

```python
import numpy as np
 
a = np.array([[1,2],[3,4],[5,6]])
 
print ('第一个数组：')
print (a)
print ('\n')
 
print ('未传递 Axis 参数。 在删除之前输入数组会被展开。')
print (np.insert(a,3,[11,12]))
print ('\n')
print ('传递了 Axis 参数。 会广播值数组来配输入数组。')
 
print ('沿轴 0 广播：')
print (np.insert(a,1,[11],axis = 0))
print ('\n')
 
print ('沿轴 1 广播：')
print (np.insert(a,1,11,axis = 1))
```

输出结果如下：

```python
第一个数组：
[[1 2]
 [3 4]
 [5 6]]


未传递 Axis 参数。 在删除之前输入数组会被展开。
[ 1  2  3 11 12  4  5  6]


传递了 Axis 参数。 会广播值数组来配输入数组。
沿轴 0 广播：
[[ 1  2]
 [11 11]
 [ 3  4]
 [ 5  6]]


沿轴 1 广播：
[[ 1 11  2]
 [ 3 11  4]
 [ 5 11  6]]
```

numpy.delete 函数返回从输入数组中删除指定子数组的新数组。 与 insert() 函数的情况一样，如果未提供轴参数，则输入数组将展开。

```python
numpy.delete(arr, obj, axis)
```

参数说明：

- `arr`：输入数组
- `obj`：可以被切片，整数或者整数数组，表明要从输入数组删除的子数组
- `axis`：沿着它删除给定子数组的轴，如果未提供，则输入数组会被展开

```python
import numpy as np
 
a = np.arange(12).reshape(3,4)
 
print ('第一个数组：')
print (a)
print ('\n')
 
print ('未传递 Axis 参数。 在插入之前输入数组会被展开。')
print (np.delete(a,5))
print ('\n')
 
print ('删除第二列：')
print (np.delete(a,1,axis = 1))
print ('\n')
 
print ('包含从数组中删除的替代值的切片：')
a = np.array([1,2,3,4,5,6,7,8,9,10])
print (np.delete(a, np.s_[::2]))
```

输出结果为

```python
第一个数组：
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]


未传递 Axis 参数。 在插入之前输入数组会被展开。
[ 0  1  2  3  4  6  7  8  9 10 11]


删除第二列：
[[ 0  2  3]
 [ 4  6  7]
 [ 8 10 11]]


包含从数组中删除的替代值的切片：
[ 2  4  6  8 10]
```



# Pandas

## Series

Series 由索引（index）和列组成，函数如下：

```python
pandas.Series( data, index, dtype, name, copy)
```

参数说明：

- **data**：一组数据(ndarray 类型)。
- **index**：数据索引标签，如果不指定，默认从 0 开始。
- **dtype**：数据类型，默认会自己判断。
- **name**：设置名称。
- **copy**：拷贝数据，默认为 False。

```python
import pandas as pd
a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar)
print(myvar["y"])

# 输出
# x    Google
# y    Runoob
# z      Wiki
# dtype: object
# Runoob
```

也可以使用 key/value 对象，类似字典来创建 Series：

```python
import pandas as pd
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites)
print(myvar)

# 输出
# 1    Google
# 2    Runoob
# 3      Wiki
# dtype: object
```

如果我们只需要字典中的一部分数据，只需要指定需要数据的索引即可，如下实例：

```python
import pandas as pd
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites, index = [1, 2])
print(myvar)

# 输出
# 1    Google
# 2    Runoob
# dtype: object
```

## DataFrame

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

![Ds07SC](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/09/24/Ds07SC.jpg)

DataFrame 构造方法如下：

```python
pandas.DataFrame( data, index, columns, dtype, copy)
```

参数说明：

- **data**：一组数据(ndarray、series, map, lists, dict 等类型)。
- **index**：索引值，或者可以称为行标签。
- **columns**：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
- **dtype**：数据类型。
- **copy**：拷贝数据，默认为 False。

```python
import pandas as pd
data = [['Google',10],['Runoob',12],['Wiki',13]]
df = pd.DataFrame(data,columns=['Site','Age'],dtype=float)
print(df)

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}
df = pd.DataFrame(data)
print(df)

# 输出
#      Site   Age
# 0  Google  10.0
# 1  Runoob  12.0
# 2    Wiki  13.0
```

还可以使用字典（key/value），其中字典的 key 为列名:

```python
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print (df)

# 输出
#    a   b     c
# 0  1   2   NaN
# 1  5  10  20.0
# 没有对应的部分数据为 NaN
```

Pandas 可以使用 **loc** 属性返回指定行的数据，如果没有设置索引，第一行索引为 **0**，第二行索引为 **1**，以此类推：

```python
import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
# 返回第一行
print(df.loc[0])
# 返回第多行数据，使用[[]]格式
print(df.loc[[0,1]])

# 指定索引
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df.loc["day2"])
```

**注意：**返回结果其实就是一个 Pandas DataFrame 数据。



**. loc是按照行名列名进行访问，iloc按照位置进行访问。**

```python
res1 = df.loc['row1','col1']
print(res1)

res2 = df.iloc[0,0]
print(res2)

# 返回都是0行0列的元素
```

## CSV

读取csv和存储为csv

```python
df = pd.read_csv('nba.csv')
df.to_csv('nba.csv')
```

输出处理

```python
df = pd.read_csv('nba.csv')

# 读取前面的n行，默认返回5行
print(df.head(10))

# 结尾n行，默认5行
print(df.tail())

# 返回表格的一些基本信息
print(df.info()) 
```

## JSON

JSON是存储和交换文本信息的语法，类似 XML，但JSON 比 XML 更小、更快，更易解析

```json
[
   {
   "id": "A001",
   "name": "菜鸟教程",
   "url": "www.runoob.com",
   "likes": 61
   },
   {
   "id": "A002",
   "name": "Google",
   "url": "www.google.com",
   "likes": 124
   },
   {
   "id": "A003",
   "name": "淘宝",
   "url": "www.taobao.com",
   "likes": 45
   }
]
```

```python
# 读取json文件
df = pd.read_json('sites.json')
# 从URL中读取JSON数据
URL = 'https://static.runoob.com/download/sites.json'
df = pd.read_json(URL)
```

## 数据清洗

### 清洗空值

如果我们要删除包含空字段的行，可以使用 **dropna()** 方法，语法格式如下：

```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
```

**参数说明：**

- **axis**：默认为 **0**，表示逢空值剔除整行，如果设置参数 **axis＝1** 表示逢空值去掉整列。
- **how**：默认为 **'any'** 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 **how='all'** 一行（或列）都是 NA 才去掉这整行。
- thresh：设置需要多少非空值的数据才可以保留下来的。
- subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
- inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。



使用`isnull()`判断各个单元格是否为空

```python
import pandas as pd

df = pd.read_csv('property-data.csv')

print (df['NUM_BEDROOMS'])
# 为空返回true，不为空返回false
print (df['NUM_BEDROOMS'].isnull())
```

删除包含空数据的行

```py
import pandas as pd

df = pd.read_csv('property-data.csv')
new_df = df.dropna()
print(new_df.to_string())
```

**注意：**默认情况下，dropna() 方法返回一个新的 DataFrame，不会修改源数据，如果需要修改原数据可以设置属性`inplace = True`

也可以移除指定列有空值的行：

```python
import pandas as pd

df = pd.read_csv('property-data.csv')
# 如果列名“ST_NUM”存在空值则删除行
df.dropna(subset=['ST_NUM'], inplace = True)
print(df.to_string())
```

也可以使用 `fillna()` 方法来替换一些空字段：

```python
import pandas as pd

df = pd.read_csv('property-data.csv')
# 使用12345替换空字段
df.fillna(12345, inplace = True)
print(df.to_string())
```

可以指定列进行替换

```python
import pandas as pd

df = pd.read_csv('property-data.csv')
# 使用12345替换列“PID”中为空的数据
df['PID'].fillna(12345, inplace = True)
print(df.to_string())
```

实际处理过程中常常替换为列的均值`mean()`、中位数`median()`或众数`mode()`

```py
import pandas as pd

df = pd.read_csv('property-data.csv')
# 使用列的均值来替换空值
x = df["ST_NUM"].mean()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())
```

### 清洗格式错误的数据

格式化日期：

```python
import pandas as pd

# 第三个日期格式错误
data = {
  "Date": ['2020/12/01', '2020/12/02' , '20201226'],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

df['Date'] = pd.to_datetime(df['Date'])

print(df.to_string())

# 输出
#            Date  duration
# day1 2020-12-01        50
# day2 2020-12-02        40
# day3 2020-12-26        45
```

### 清洗数值错误数据

可以将错误数据的行删除：

```python
import pandas as pd

person = {
  "name": ['Google', 'Runoob' , 'Taobao'],
  "age": [50, 40, 12345]    # 12345 年龄数据是错误的
}

df = pd.DataFrame(person)

for x in df.index:
  if df.loc[x, "age"] > 120:
    df.drop(x, inplace = True)

print(df.to_string())
```

### 清洗重复数据

可以使用 `duplicated() `和 `drop_duplicates() `方法。

如果对应的数据是重复的，`duplicated() `会返回 True，否则返回 False。

```python
import pandas as pd

person = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]  
}
df = pd.DataFrame(person)
print(df.duplicated())

# 输出
# 0    False
# 1    False
# 2     True
# 3    False
# dtype: bool
```

删除数据使用`drop_duplicates()`

```python
import pandas as pd

persons = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]  
}
df = pd.DataFrame(persons)
df.drop_duplicates(inplace = True)
print(df)

# 输出
#      name  age
# 0  Google   50
# 1  Runoob   40
# 3  Taobao   23
```

# matplotlib

## Pyplot

```python
# 画单条线
plot([x], y, [fmt], *, data=None, **kwargs)
# 画多条线
plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
```
参数说明：

- **x, y：**点或线的节点，x 为 x 轴数据，y 为 y 轴数据，数据可以列表或数组。
- **fmt：**可选，定义基本格式（如颜色、标记和线条样式）。 
  - 颜色字符：'b' 蓝色，'m' 洋红色，'g' 绿色，'y' 黄色，'r' 红色，'k' 黑色，'w' 白色，'c' 青绿色。多条曲线不指定颜色时，会自动选择不同颜色。
  - 线型参数：'‐' 实线，'‐‐' 破折线，'‐.' 点划线，':' 虚线。
  - 标记字符：'.' 点标记，',' 像素标记(极小点)，'o' 实心圈标记，'v' 倒三角标记，'^' 上三角标记，'>' 右三角标记，'<' 左三角标记...等等。
- ***\*kwargs：**可选，用在二维平面图上，设置指定属性，如标签，线的宽度等。

```python
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.plot(xpoints, ypoints)
plt.show()
```

## 绘图标记

如果想要给坐标自定义一些不一样的标记，就可以使用 **plot()** 方法的 **marker** 参数来定义。
```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([1,3,4,5,8,9,6,1,3,4,5,2,4])

plt.plot(ypoints, marker = 'o')
plt.show()
```

![img](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/09/25/pl_marker01.png)

## 绘图线

线的宽度可以使用 **linewidth** 参数来定义，简写为 **lw**，值可以是浮点数，如：**1**、**2.0**、**5.67** 等。

```python
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([6, 2, 13, 10])

plt.plot(ypoints, linewidth = '12.5')
plt.show()
```

![img](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/09/25/pl_line-6.png)

## 轴标签和标题

使用 `xlabel()` 和 `ylabel() `方法来设置 x 轴和 y 轴的标签。

使用`title()` 方法来设置标题。

三个方法都提供了`loc`参数来控制显示的位置，可以设置为: **'left', 'right', 和 'center'， 默认值为 'center'**。

## 网格线

```python
matplotlib.pyplot.grid(b=None, which='major', axis='both', )
```

**参数说明：**

- **b**：可选，默认为 None，可以设置布尔值，true 为显示网格线，false 为不显示，如果设置 **kwargs 参数，则值为 true。
- **which**：可选，可选值有 'major'、'minor' 和 'both'，默认为 'major'，表示应用更改的网格线。
- **axis**：可选，设置显示哪个方向的网格线，可以是取 'both'（默认），'x' 或 'y'，分别表示两个方向，x 轴方向或 y 轴方向。
- ***\*kwargs**：可选，设置网格样式，可以是 color='r', linestyle='-' 和 linewidth=2，分别表示网格线的颜色，样式和宽度。

## figure

前面都是使用plot快速生成图像，但如果使用面向对象的编程思想，我们就可以更好地控制和自定义图像。

Matplotlib 提供了`matplotlib.figure`图形类模块，它包含了创建图形对象的方法。

```python
from matplotlib import pyplot as plt
#创建图形对象
fig = plt.figure()
```

参数设置：

- figsize：指定画布的大小，(宽度,高度)，单位为英寸。
- dpi：指定绘图对象的分辨率，即每英寸多少个像素，默认值为80。
- facecolor：背景颜色
- dgecolor：边框颜色
- frameon：是否显示边框。

## axes

Matplotlib 定义了一个 axes 类（轴域类），该类的对象被称为 axes 对象（即轴域对象），它指定了一个有数值范围限制的绘图区域。在一个给定的画布（figure）中可以包含多个 axes 对象，但是同一个 axes 对象只能在一个画布中使用。

通过调用 `add_axes()` 方法能够将 axes 对象添加到画布中，该方法用来生成一个 axes 轴域对象，对象的位置由参数`rect`决定。

rect 是位置参数，接受一个由 4 个元素组成的浮点数列表，形如 [left, bottom, width, height] ，它表示添加到画布中的矩形区域的左下角坐标(x, y)，以及宽度和高度。

```python
ax=fig.add_axes([0.1,0.1,0.8,0.8])
```

注意：每个元素的值是画布宽度和高度的分数。即将画布的宽、高作为 1 个单位。比如，[ 0.1, 0.1, 0.8, 0.8]，它代表着从画布 10% 的位置开始绘制, 宽高是画布的 80%。

它有以下成员函数：

- legend()

  ```python
  ax.legend(handles, labels, loc)
  ```

  - labels 是一个字符串序列，用来指定标签的名称；
  - loc 是指定图例位置的参数，其参数值可以用字符串或整数来表示；
  - handles 参数，它也是一个序列，它包含了所有线型的实例；

- plot()

```python
from matplotlib import pyplot as plt
import numpy as np
import math
x = np.arange(0, math.pi*2, 0.05)
y = np.sin(x)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)
ax.set_title("sine wave")
ax.set_xlabel('angle')
ax.set_ylabel('sine')
plt.show()
```

## 绘制多图

使用 pyplot 中的 `subplot() `和 `subplots() `方法来绘制多个子图。

```python
subplot(nrows, ncols, index, **kwargs)
```

以上函数将整个绘图区域分成 nrows 行和 ncols 列，然后从左到右，从上到下的顺序对每个子区域进行编号 **1...N** ，左上的子区域的编号为 1、右下的区域编号为 N，编号可以通过参数 **index** 来设置。

```python
#plot 1:
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])

plt.subplot(1, 2, 1)
plt.plot(xpoints,ypoints)
plt.title("plot 1")

#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("plot 2")
```

---

`matplotlib.pyplot`模块提供了一个 subplots() 函数，它的使用方法和 subplot() 函数类似。其不同之处在于，subplots() 既创建了一个包含子图区域的画布，又创建了一个 figure 图形对象，而 subplot() 只是创建一个包含子图区域的画布。

函数的返回值是一个元组，包括一个图形对象和所有的 axes 对象。其中 axes 对象的数量等于 nrows * ncols，且每个 axes 对象均可通过索引值访问（从1开始）。

```python
import matplotlib.pyplot as plt
fig,a =  plt.subplots(2,2)
import numpy as np
x = np.arange(1,5)
#绘制平方函数
a[0][0].plot(x,x*x)
a[0][0].set_title('square')
#绘制平方根图像
a[0][1].plot(x,np.sqrt(x))
a[0][1].set_title('square root')
#绘制指数函数
a[1][0].plot(x,np.exp(x))
a[1][0].set_title('exp')
#绘制对数函数
a[1][1].plot(x,np.log10(x))
a[1][1].set_title('log')
plt.show()
```

## 散点图

使用 pyplot 中的 scatter() 方法来绘制散点图。

```python
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
```

**参数说明：**

- **x，y**：长度相同的数组，也就是我们即将绘制散点图的数据点，输入数据。

- **s**：点的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小。

- **c**：点的颜色，默认蓝色 'b'，也可以是个 RGB 或 RGBA 二维行数组。

- **marker**：点的样式，默认小圆圈 'o'。

- **cmap**：Colormap，默认 None，标量或者是一个 colormap 的名字，只有 c 是一个浮点数数组的时才使用。如果没有申明就是 image.cmap。

- **norm**：Normalize，默认 None，数据亮度在 0-1 之间，只有 c 是一个浮点数的数组的时才使用。

- **vmin，vmax：**亮度设置，在 norm 参数存在时会忽略。

- **alpha：**透明度设置，0-1 之间，默认 None，即不透明。

- **linewidths：**标记点的长度。

- **edgecolors：**颜色或颜色序列，默认为 'face'，可选值有 'face', 'none', None。

- **plotnonfinite：**布尔值，设置是否使用非限定的 c ( inf, -inf 或 nan) 绘制点。

- **kwargs：**其他参数。

## 柱状图

使用 pyplot 中的 bar() 方法来绘制柱形图。

```python
matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
```

**参数说明：**

- **x**：浮点型数组，柱形图的 x 轴数据。

- **height**：浮点型数组，柱形图的高度。

- **width**：浮点型数组，柱形图的宽度，默认0.8。

- **bottom**：浮点型数组，底座的 y 坐标，默认 0。

- **align**：柱形图与 x 坐标的对齐方式，'center' 以 x 位置为中心，这是默认值。 'edge'：将柱形图的左边缘与 x 位置对齐。要对齐右边缘的条形，可以传递负数的宽度值及 align='edge'。

- **kwargs：**其他参数。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])

plt.bar(x,y)
plt.show()
```

垂直方向的柱形图可以使用 **barh()** 方法来设置：

## 饼图

使用 pyplot 中的 pie() 方法来绘制饼图。

```python
matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=0, 0, frame=False, rotatelabels=False, *, normalize=None, data=None)[source]
```

**参数说明：**

- **x**：浮点型数组，表示每个扇形的面积。

- **explode**：数组，表示各个扇形之间的间隔，默认值为0。

- **labels**：列表，各个扇形的标签，默认值为 None。

- **colors**：数组，表示各个扇形的颜色，默认值为 None。

- **autopct**：设置饼图内各个扇形百分比显示格式，**%d%%** 整数百分比，**%0.1f** 一位小数， **%0.1f%%** 一位小数百分比， **%0.2f%%** 两位小数百分比。

- **labeldistance**：标签标记的绘制位置，相对于半径的比例，默认值为 1.1，如 **<1**则绘制在饼图内侧。

- **pctdistance：**：类似于 labeldistance，指定 autopct 的位置刻度，默认值为 0.6。

- **shadow：**：布尔值 True 或 False，设置饼图的阴影，默认为 False，不设置阴影。

- **radius：**：设置饼图的半径，默认为 1。

- **startangle：**：起始绘制饼图的角度，默认为从 x 轴正方向逆时针画起，如设定 =90 则从 y 轴正方向画起。

- **counterclock**：布尔值，设置指针方向，默认为 True，即逆时针，False 为顺时针。

- **wedgeprops** ：字典类型，默认值 None。参数字典传递给 wedge 对象用来画一个饼图。例如：wedgeprops={'linewidth':5} 设置 wedge 线宽为5。

- **textprops** ：字典类型，默认值为：None。传递给 text 对象的字典参数，用于设置标签（labels）和比例文字的格式。

- **center** ：浮点类型的列表，默认值：(0,0)。用于设置图标中心位置。

- **frame** ：布尔类型，默认值：False。如果是 True，绘制带有表的轴框架。

- **rotatelabels** ：布尔类型，默认为 False。如果为 True，旋转每个 label 到指定的角度。

```python
import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])

plt.pie(y,
        labels=['A','B','C','D'], # 设置饼图标签
        colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"], # 设置饼图颜色
        explode=(0, 0.2, 0, 0), # 第二部分突出显示，值越大，距离中心越远
        autopct='%.2f%%', # 格式化输出百分比
       )
plt.title("RUNOOB Pie Test")
plt.show()
```

