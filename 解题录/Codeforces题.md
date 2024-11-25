### 模板
link：

思路：

####代码
```python

```
学习总结：

### CF455A: Boredom
link：https://codeforces.com/contest/455/problem/A （dp）

思路：

是dp题，状态转移方程是dp[i]=max(dp[i-1],dp[i-2]+arr_2[i]*i)，即要么选该数字，包含该数字-2的（题目规定不能前一），要么不选就维持分数），要注意的是一开始作dp数组时需要使用max(arr_1)+1,如果直接用n，n如果很大会re，dp数组长度最大只需要到数列中最大的号码即可

####代码
```python
n=int(input())
arr_1=list(map(int,input().split()))
arr_2=[0]*(max(arr_1)+1)
for i in arr_1:
    arr_2[i]+=1
dp=[0]*(max(arr_1)+1)
dp[1]=arr_2[1]
 
for i in range(2,max(arr_1)+1):
    dp[i]=max(dp[i-1],dp[i-2]+arr_2[i]*i)
 
print(dp[max(arr_1)])
```
学习总结：

dp题大部分都是需要用max函数比较，且需要一个最基础的dp[0]/[1]


### 545C. Woodcutters
link：https://codeforces.com/problemset/problem/545/C （greedy/dp）

思路：

判断能不能向左或右倒，且如果向右倒，下一棵树要留意此距离


####代码
```python
n = int(input())
tree = []


for _ in range(n):
    a, b = map(int, input().split())
    tree.append((a, b))

prv = float('-inf')
cnt = 0

for i in range(n):

    if prv + tree[i][1] < tree[i][0]: #左边能够倒下
        cnt += 1
        prv = tree[i][0]

    elif i < n - 1 and tree[i][1] + tree[i][0] < tree[i+1][0]:#能够向右倒
        cnt += 1
        prv = tree[i][0]+tree[i][1]
    elif i==n-1:
        cnt+=1
    else:
        prv = tree[i][0]

print(cnt)
```
学习总结：

学会判断情况，根据情景写出相应代码


### 158B. Taxi
link：https://codeforces.com/problemset/problem/158/B （greedy）

思路：

已备注在代码里，清晰可知

####代码
```python
import math
n = int(input())
arr_1 = list(map(int, input().split()))

car = 0
sum1 = sum2 = sum3 = 0

for i in arr_1:
    if i == 4:
        car += 1  # 每个4的人都需要一个车
    elif i == 3:
        sum3 += 1  # 记录3的人数
    elif i == 2:
        sum2 += 1  # 记录2的人数
    else:
        sum1 += 1  # 记录1的人数

# 先处理3的人和1的人
min_3_and_1 = min(sum3, sum1)
car += sum3  # 3的数量都需要一个车
sum1 -= min_3_and_1  # 减去与3配对的1

# 处理2的人，每两个2可以拼成一个车
car += sum2 // 2
if sum2 % 2:  # 如果还有一个剩余的2
    car += 1
    sum1 = max(0, sum1 - 2)  # 如果有剩余的1，减去最多2个

# 剩下的1的人，每4个1可以拼成一个车
car += math.ceil(sum1 / 4 ) #向上取整

print(car)
```
学习总结：

这种题目如同装箱子，要懂得如何分配每组，多少+1，建议可以去看装箱子题，比这题略难


### 230B. T-primes
link：http://codeforces.com/problemset/problem/230/B （binary search, implementation, math, number theory)

思路：

只要一个数的因数有三个就是Tprime，且除去1和本身，另一个因数必为Tprime本身的平方根且必是质数
埃氏筛查法原理
1. 初始化一个列表 `is_prime`，假设所有数都是质数。
2. 标记 0 和 1 不是质数。
3. 从 2 开始，依次遍历每个数：
    - 如果该数是质数，则将它的所有倍数标记为非质数（从 `i^2` 开始）。
4. 返回标记完的质数表。


####代码
```python
import math

# 使用埃氏筛法生成所有小于等于 10^6 的质数
def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False  # 0 和 1 不是质数
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return is_prime

# 预处理所有小于等于 10^6 的质数
MAX_LIMIT = 10**6
is_prime = sieve_of_eratosthenes(MAX_LIMIT)

n = int(input())  # 输入数组元素个数
arr = list(map(int, input().split()))  # 输入数组

for i in arr:
    sqrt_i = math.isqrt(i)  # 获取整数平方根
    if sqrt_i * sqrt_i == i and is_prime[sqrt_i]:  # 判断是否为完全平方数且平方根为质数
        print("YES")
    else:
        print("NO")

```
学习总结：

多读数学方法可能有新收获！


### 270A. Fancy Fence
link：https://codeforces.com/contest/270/problem/A （math）

思路：

题目给定的是转角（即外角），而正多边形的外角为180∘−内角，所以可以通过判断是否存在一个正整数n，使得360∘/n=a

####代码
```python
t=int(input())
for _ in range(t):
    angle=int(input())
    if 360%(180-angle)==0:
        print('YES')
    else:
        print('NO')
```
学习总结：

你学废了吗？


### 71A. Way Too Long Words
link：https://codeforces.com/contest/71/problem/A （strings）

思路：

####代码
```python
n=int(input())
for _ in range(n):
    word=input()
    if len(word)<=10:
        print(word)
    else:
        print(f"{word[0]}{len(word)-2}{word[len(word)-1]}")
```
学习总结：

别被题目唬住！

### 122A. Lucky Division
link：https://codeforces.com/contest/122/problem/A （brute force, number theory）

思路：

(char in '47' for char in lucky) 是一个生成器表达式，它会逐个检查字符串 lucky 中的每个字符，判断它是否在字符串 '47' 中。

all() 返回 True，当且仅当可迭代对象中的所有元素都为 True。如果有一个元素为 False，则 all() 返回 False。

####代码
```python
def check_luck(luck):
    return all(char in '47' for char in luck)
flag=0
lucky=input()
if check_luck(lucky):
    print("YES")
    flag=1
else:
    for i in range(1,int(lucky)//2+1):
        if check_luck(str(i)) and int(lucky)%i==0:
            print("YES")
            flag=1
            break

if flag==0:
    print("NO")
```
学习总结：

学习all（）和 <bool> for _ in _ 用法


### 118A. String Task
link：https://codeforces.com/contest/118/problem/A （strings）

思路： 

####代码
```python
# 获取用户输入
arr1 = input().lower()

# 去除元音字母
vowels = "aeiouy"
for vowel in vowels:
    arr1 = arr1.replace(vowel, '')

# 在每个字符前加上点
arr1 = '.' + '.'.join(arr1)

# 输出结果
print(arr1)

```
学习总结：

玩弄语法！


### 723A. The New Year: Meeting Friends
link：https://codeforces.com/contest/723/problem/A

思路：

顺序排列，以中间人所在地为目标是最佳的，那么只需要两个人移动即可

####代码
```python
import math
three=list(map(int,input().split()))
three=sorted(three)
result=abs(three[0]-three[1])+abs(three[2]-three[1])
print(result)
```
学习总结：

无


### 615A. Bulbs
link：https://codeforces.com/problemset/problem/615/A

思路：

看代码即可理解

####代码
```python
n,m=map(int,input().split())
count=[]*m
for _ in range(n):
    x=list(map(int,input().split()))
    for i in range(1,len(x)):
        if x[i] not in count:
            count.append(x[i])

if len(count)==m:
    print("YES")
else:
    print("NO")
```
学习总结：

无

### 705A. Hulk
link：https://codeforces.com/contest/705/problem/A

思路：

注意有一个it在最后一个字

####代码
```python
layer1="I hate"
layer2='I love'

n=int(input())
output=layer1
for i in range(1,n):
    if i%2==0:
        output+=' that '+layer1
    else:
        output+=' that '+layer2
output+=' it'
print(output)
```
学习总结：

I hate that I love that I hate that I love that ...


### 282A. Bit++
link：https://codeforces.com/problemset/problem/282/A

思路：

####代码
```python
n=int(input())
x=0
for _ in range(n):
    count=input()
    if '+' in count:
        x+=1
    else:
        x-=1
print(x)
```
学习总结：

机考按这个来多好

### 339A. Helpful Maths
link：https://codeforces.com/contest/339/problem/A

思路：

先去掉符号，排序，再加上符号

####代码
```python
sum_1=input()
if len(sum_1)==1:
    print(sum_1)
else:
    sum_1=sum_1.replace('+','')
    sort_sum=sorted(sum_1)
    sorted_sum='+'.join(sort_sum)
    print(sorted_sum)
```
学习总结：

三年级怎么就不会微积分？（开玩笑

### 69A. Young Physicist
link：https://codeforces.com/problemset/problem/69/A

思路：

受力平衡就是三个方向合力为0

####代码
```python
n = int(input())  # Number of vectors
vect = []  # List to store vectors

# Reading vectors from input
for _ in range(n):
    vect.append(list(map(int, input().split())))

sumx, sumy, sumz = 0, 0, 0  # Initializing sums for x, y, z

# Summing up the x, y, z components
for i in range(n):
    sumx += vect[i][0]
    sumy += vect[i][1]
    sumz += vect[i][2]

# Checking if the resultant vector is a zero vector
if sumx == sumy == sumz == 0:
    print("YES")
else:
    print("NO")
```
学习总结：

死去的物理别复活！


### 58A. Chat room
link：https://codeforces.com/problemset/problem/58/A

思路：

看注释，善用 for in

####代码
```python
def can_say_hello(s):
    target = "hello"
    index = 0

    # 遍历输入的字符串 s
    for char in s:
        # 检查当前字符是否匹配目标字符串 target 中的当前字符
        if char == target[index]:
            index += 1
        # 如果已经找到了 "hello" 的所有字符
        if index == len(target):
            return "YES"

    # 如果遍历完字符串后没有找到完整的 "hello"
    return "NO"


# 输入处理
s = input()
print(can_say_hello(s))

```
学习总结：
无

### 580A. Kefa and First Step
link：https://codeforces.com/problemset/problem/580/A

思路：

一旦出现上升情况cnt归0

####代码
```python
n=int(input())
arr=list(map(int,input().split()))
cnt=0
ans=[]

if len(arr)==1:
    print('1')
else:
    for i in range(0,len(arr)-1):
        if arr[i]<=arr[i+1]:
            cnt+=1
        elif arr[i]>arr[i+1]:
            cnt=0
        ans.append(cnt)
    print(max(ans)+1)

```
学习总结：
无

### 474A. Keyboard
link：https://codeforces.com/contest/474/problem/A

思路：

list出一个键盘，方便移动操作，如果R那么就按照键盘上一位输出（根据索引）

####代码
```python
keyboard=['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l',';','\'','z','x','c','v','b','n','m',',','.','/']
dir=input()
line=list(input())
outP=[]
if dir=='R':
   for i in line:
       if i in keyboard:
           outP.append(keyboard[keyboard.index(i)-1])
elif dir=='L':
   for i in line:
       if i in keyboard:
           outP.append(keyboard[keyboard.index(i)+1])

print(''.join(outP))
```
学习总结：

.index(i)求i的索引


### 460A. Vasya and Socks
link：https://codeforces.com/contest/460/problem/A

思路：

在m倍天不消耗袜子

####代码
```python
n,m=map(int,input().split())
day=0
while n!=0:
    day+=1
    if day%m==0:
        continue
    n-=1
print(day)
```
学习总结：


### 1374B - Multiply by 2, divide by 6
link：https://codeforces.com/problemset/problem/1374/B

思路：

 #如果能成功，把n想成2^x 3^y，的乘积，同时能被6整除或者乘2了能被6整除

####代码
```python
t=int(input())
for _ in range(t):
    n=int(input())
    cnt=0
    while n!=1:
        if n%2==0 and n%3==0:  
          n=n//6
          cnt+=1
        elif n%2!=0 and n%3==0:
           n=n*2
           cnt+=1
        else:
            print("-1")
            break
    else:print(cnt)
```
学习总结：

此题需要一定数学逻辑思考！

### B. Restore the Weather
link：https://codeforces.com/problemset/problem/1833/B

思路：

排序的原因是，为了尽量减小差值，较大的 `b` 应该匹配较大的 `a`

####代码
```python
t = int(input())
for _ in range(t):
    j, k = map(int, input().split())

    l1 = list(map(int, input().split()))
    v = [(l1[i], i) for i in range(j)]
    v.sort() 

    l2 = list(map(int, input().split()))
    l2.sort()

    z = [0] * j
    for i in range(j):
        z[v[i][1]] = l2[i] 

    for data in z:
        print(data, end=" ")
    print()
```
学习总结：

有点烧脑，但是拿笔一写，还是烧脑，原来是自己写不清楚！

### 1879B. Chips on the Board
link：https://codeforces.com/contest/1879/problem/B

思路：

行最小数去加列 比较 列最小数加行，去最小即可

####代码
```python
t = int(input())
for _ in range(t):
    n = int(input())
    *a, = map(int, input().split())
    *b, = map(int, input().split())
    
    min_a = min(a)
    min_b = min(b)
    
    ans1 = sum([min_a + i for i in b])
    ans2 = sum([min_b + i for i in a])
    print(min(ans1, ans2))
```
学习总结：

动手动脑动笔！

### 模板
link：

思路：

####代码
```python

```
学习总结：

### 模板
link：

思路：

####代码
```python

```
学习总结：

### 模板
link：

思路：

####代码
```python

```
学习总结：

### 模板
link：

思路：

####代码
```python

```
学习总结：

### 模板
link：

思路：

####代码
```python

```
学习总结：

### 模板
link：

思路：

####代码
```python

```
学习总结：
