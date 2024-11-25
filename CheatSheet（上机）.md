## 一、有关语法

### 输入
```n = int(input())```:输入 n为数字

```list1 = list(map(int, input().split()))``` ：输入一行形成列表 其中```input().split()```将整行输入按空格分割为多个部分 

```M, N = map(int, input().split())```一次读取两个数（经典必会）

### 输出
```print("{a:._f}"))```：输出a保留_位小数

```print(*arr)```：列表中的所有元素展开为独立的参数，方便输出

```print(,end='')```： 就是输出后不要换下一行(也可用于输出后添加东西在后边）

```print(f"{}")```：可用于输出一些符号代表数值同时又不会有空格

```print(, sep='\n')```：输出每个元素占一行


### 函数（简单的不收录）

import math -> ```math.ceil()``` :向上取整

```abs()```：绝对值

```ord()```： 是转变成可比较数（但不同于int)

```chr()```： 转变成字符

 ```a= float("inf")```：inf无穷，需用float(也可以加负号变成负无穷）```float('-inf')```

 ```.replace(" ",' ')```：将字符串中的某些字符替换（前为欲替换字符，后为替换后）

 ```
try:
    while True:
        
except EOFError:
    pass/break  # This will stop the loop when no more input is given
```

```''.join（）```：以''链接（）内的元素

```all()```：all() 返回 True，当且仅当可迭代对象中的所有元素都为 True。如果有一个元素为 False，则 all() 返回 False

```[::-1]```意思是翻转

```bin()``` 是转换为二进制，但是开头会带有多余的 '0b' 

```global ```：使全局变量

```from collections import deque -> deque()```：双端队列，可以在两端高效地进行插入和删除操作的数据结构<br> 
**左侧操作：**<br>
```appendleft(x)```: 在左侧插入元素 x。<br>
```popleft()```: 从左侧弹出并返回一个元素。<br>
**右侧操作：**<br>
```append(x)```: 在右侧插入元素 x（类似于列表的 append）。<br>
```pop()```: 从右侧弹出并返回一个元素。

### 数组list用
```.index()```：找出某个值（数组内）的索引值

```.append()```：后面添加元素

```.sort()```：排序数组内元素 ；```.sort(reverse=True)```：逆序排列 ；```sorted(arr_1, key=lambda x: )```，将arr_1按后边的指示排序（也可以是处理后排序）；```sorted(,key=lambda x:( , ))```用来面对当数值/字符无法按此处理时用‘，’后的方法排序

```.rstrip()```： 表示移除字符串末尾（右侧）的指定字符（默认为空格和换行符）

```arr=[(arr1[i], i) for i in range(j)] ```数组内创造tuple（带index）

```arr = [[0] * bc for _ in range(ar)]``` 是用来初始化一个二维列表（矩阵）的，它的作用是创建一个大小为 ar 行、bc 列的矩阵，并将所有元素初始化为 0(也可以没有元素）。

```arr[-1]```：arr中最后一个元素

```arr.find(' ' ,0) ```返回索引，找不到返回-1，后面数字可以指定从第几个索引以后开始找

```list_n.remove(i)```：去除i

```arr1=arr2.copy()```将arr2内容copy给arr1


### 字典
写法：```dict1={'a':0,'b':1...}```; 调用方式 ```dict1['a']=0```


## 二、算法或经典方法

