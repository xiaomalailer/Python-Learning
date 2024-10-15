# Assignment #4: T-primes + 贪心


2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



思路：
要稳赚不陪，那就买负数的，先做排序（可能也可以不），然后把负数的加起来就好了（耗时7分钟）



代码

```python
# 
n,m=map(int,input().split())
arr_1=list(map(int,input().split()))
arr_1=sorted(arr_1)
earn=0
for i in range(m):
    if arr_1[i]>0:
        break
    earn-=arr_1[i]
print(abs(earn))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20171045.png?raw=true)




### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A

思路：
要赚到就从大币值开始拿起，一旦拿到恰好超过剩余的就停手（耗时5分04秒）


代码

```python
n=int(input())
arr_1=list(map(int,input().split()))
c=sum(arr_1)
b=0
coin=0
arr_1=sorted(arr_1,reverse=True)
for i in arr_1:
    b+=i
    coin+=1
    if b>c//2:
        break
print(coin)
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20172124.png?raw=true)




### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B

思路：

花了大把时间尝试读懂题目，但是发现去读测试数据更好，这题是在说创造一个n*n方格后，让每行/每列都有一个chips的amount，所以有两种情况，一种是行里最小的数去加每列的数 ，和列里最小的去加每行的数，比较即可得

（耗时15分43秒）

代码

```python
t=int(input())
for _ in range(t):
    n=int(input())
    a=list(map(int,input().split()))
    b=list(map(int,input().split()))
    min_a=min(a)
    min_b=min(b)
    asw_a=sum([min_a+i for i in b])
    asw_b=sum([min_b+i for i in a])
    print(min(asw_a,asw_b))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20174337.png?raw=true)




### 158B. Taxi

*special problem, greedy, implementation, 1100, https://codeforces.com/problemset/problem/158/B

思路：

应该说多亏了CF能看测试数据我才能一步步查漏补缺，所以写得很繁琐，基本上想法是如果一组人数是3或4就要一辆，如果一组2的成对就+1辆车，剩下的一组2人的就要一辆，然后1人的就看3人和2人那边能不能塞下

(耗时28分钟，等待测试蛮耗时)

代码

```python
import math
n=int(input())
arr_1=list(map(int,input().split()))
car=0
sum2=0
sum3=0
sum1=0
for i in arr_1:
    if i==4:
        car+=1
    elif i==3:
        car+=1
        sum3+=1
    elif i==2:
        sum2+=i
        if sum2==4:
            car+=1
            sum2-=4
    else:
        sum1+=i
if sum2>0:
    car+=1
sum31=sum3+sum2-sum1
if sum31<0:
    while abs(sum31)>4:
        sum31+=4
        car+=1
    if sum31<0:
        car+=1
print(car)
```

优化后如下：（已AC）
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


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20182848.png?raw=true)




### *230B. T-primes（选做）

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B

思路：

此题其实很好理解，只要一个数的因数有三个就是Tprime，且除去1和本身，另一个因数必为Tprime本身的平方根且必是质数，此题关键在于判断质数部分会TLE

个人用了埃氏筛法用以找到一定范围内所有质数以便不需要每个数都去跑一个判断质数的函数，埃氏筛法简单来说就是所有数的倍数都不会是质数，而且如果该数已经是其他数的倍数那么就不需要检测它的倍数了，据此就可以标记所有是和不是质数的数

（耗时约15分钟）

代码

```python
import math
def isprime(limit):
    is_prime=[True]*(limit+1)
    is_prime[0]=is_prime[1]=False
    for i in range(2,int(math.sqrt(limit))+1):
        if is_prime[i]:
            for j in range(i*i,limit+1,i):
                is_prime[j]=False
    return is_prime


max_lim=10**6
prime_num=isprime(max_lim)

n=int(input())
arr=list(map(int,input().split()))

for i in arr:
    sqrt_1=math.isqrt(i)
    if sqrt_1*sqrt_1==i and prime_num[sqrt_1]:
        print('YES')
    else:
        print('NO')

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20185537.png?raw=true)




### *12559: 最大最小整数 （选做）

greedy, strings, sortings, http://cs101.openjudge.cn/practice/12559

思路：

这题用了字典序比较，即逐字比较，sorted本来就可以达成，问题是有些数字不够长，比如测试数据中的‘11’和‘113’，那么就将数字都拉到至少10个字的长度（重复10次），再进行比较即可

当然ai也有提出可以用cmp_to_key方式自定义sorted的排序方式，但个人认为略显复杂？

（耗时约24分钟）

代码

```python
n=int(input())
arr_1=list(input().split())
arr_2=arr_1.copy()
max_num = ''.join(sorted(arr_1, key=lambda x: x*10, reverse=True))

min_num = ''.join(sorted(arr_2, key=lambda x: x*10))
print(max_num ,min_num)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20221323.png?raw=true)




## 2. 学习总结和收获
这次作业许多题目都是之前每日选做做过的，但是以前没做到的现在做到了，还是蛮开心的

taxi那题虽然花了很多时间，但是是有思路去做的，个人感觉和装箱问题有一点点相似，可能因为如此很快就可以想到3、4个人一辆车，然后再处理2个人和1个人的问题

Tprime那题试着把之前学的埃氏筛法亲手写了，大部分还是写到了（虽然忘了一点），感觉真的有学到

最后一题的话是问了ai才知道重复多次再比较

之前一直刷一题是一题，一直认为没什么学习进展，今天重复做回去才发现其实学了蛮多的，又觉得自己行了，继续刷题！（还未把选做题做完。。）

然后做完了作业觉得耗时不可信，毕竟做题环境相对太舒适。。下次试看到课上计时做

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>