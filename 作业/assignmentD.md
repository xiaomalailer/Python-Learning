# Assignment #D: 十全十美 

Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 02692: 假币问题

brute force, http://cs101.openjudge.cn/practice/02692

思路：

实在不知道该怎么下手，看了解答，只能说自己怎么那么蠢没想到，如果是假币，分为重和轻两种情况，重就满足放在左边up(右上），放在右边down，如果没有放就even，遂检查三次测试情况，如果满足则是假币，轻的假币类似，一个个检查每块币为假币的可能性即可

代码：

```python


def check_coin(coins):
    for coin in 'ABCDEFGHIJKL':
        if all( (coin in s[0] and s[2]=='down' ) or \
            (coin in s[1] and s[2]=='up' ) or \
                (coin not in s[0]+s[1] and s[2]=='even') for s in coins):
            print('{} is the counterfeit coin and it is {}.'.format(coin, 'light'))
            break
        if all((coin in s[0] and s[2]=='up' ) or \
            (coin in s[1] and s[2]=='down' ) or \
                (coin not in s[0]+s[1] and s[2]=='even') for s in coins):
            print('{} is the counterfeit coin and it is {}.'.format(coin, 'heavy'))
            break

n=int(input())
for _ in range(n):
    coins=[[],[],[]]
    for i in range(3):
        coins[i]=input().split()
    check_coin(coins)


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20172356.png?raw=true)




### 01088: 滑雪

dp, dfs similar, http://cs101.openjudge.cn/practice/01088

思路：

这题也是近期做过的一题，原先是写了一圈保护圈（100001)去包裹地图，以防跑出去，今天做的时候觉得毫无必要，就正常做dfs，只是说因为会TLE，所以用lru_cache

代码：

```python
from functools import lru_cache

move=[(0,1),(0,-1),(1,0),(-1,0)]

@lru_cache(None)
def dfs(x,y):
    ans=1
    for dx,dy in move:
        nx,ny=x+dx,y+dy
        if 0<=nx<R and 0<=ny<C and maze[nx][ny]<maze[x][y]:
            ans=max(ans,dfs(nx,ny)+1)
    return ans

R,C=map(int,input().split())
maze=[list(map(int,input().split())) for _ in range(R)]
res=0
for i in range(R):
    for j in range(C):
        res=max(res,dfs(i,j))
print(res)
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20162658.png?raw=true)




### 25572: 螃蟹采蘑菇

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：

一位信科学长告诉我的，别管有几个点要移动，每个点都去判定就行，果真能行，这题就变成了一题很简单的bfs，之前做过一次，还是参照着解答，写了type1 type2区分横的竖的，如今看来没必要

代码：

```python
from collections import deque

move = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def bfs(s_x1, s_y1, s_x2, s_y2):

    if not ((abs(s_x1 - s_x2) == 1 and s_y1 == s_y2) or (s_x1 == s_x2 and abs(s_y1 - s_y2) == 1)):
        return False

    q = deque()
    q.append((s_x1, s_y1, s_x2, s_y2))
    inq = set()
    inq.add((s_x1, s_y1, s_x2, s_y2))

    while q:
        x1, y1, x2, y2 = q.popleft()

        if maze[x1][y1] == 9 or maze[x2][y2] == 9:
            return True

        for dx, dy in move:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            if 0 <= nx1 < n and 0 <= ny1 < n and 0 <= nx2 < n and 0 <= ny2 < n:
                if maze[nx1][ny1] != 1 and maze[nx2][ny2] != 1:
                    if (nx1, ny1, nx2, ny2) not in inq:
                        inq.add((nx1, ny1, nx2, ny2))
                        q.append((nx1, ny1, nx2, ny2))

    return False



n = int(input())
maze = [list(map(int, input().split())) for _ in range(n)]
a = []
for i in range(n):
    for j in range(n):
        if maze[i][j] == 5:
            a.append([i, j])


if len(a) == 2:
    result = bfs(a[0][0], a[0][1], a[1][0], a[1][1])
    print('yes' if result else 'no')
else:
    print('no')

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20161344.png?raw=true)




### 27373: 最大整数

dp, http://cs101.openjudge.cn/practice/27373/

思路：

主要是str和int转换导致debug了些时间，这题还算理解起来不难，就是小偷背包，但是写代码前面有点难的是要怎样排列，看了解答才想到根据首位数字去排列


代码：

```python

def c(s):
    if s=='':
        return 0
    else:
        return int(s)

m=int(input()) #位数
n=int(input()) #整数数量
arr=input().split()

#按首位数字大小逆序排列
for i in range(n):
    for j in range(n-i-1):
        if arr[j]+arr[j+1]>arr[j+1]+arr[j]:
            temp=arr[j]
            arr[j]=arr[j+1]
            arr[j+1]=temp
weight=[]
for i in arr:
    weight.append(len(i))

dp=[['']*(m+1) for _ in range(n+1)]# dp[i][j] i for位数 j for 整数

for s in range(1,n+1): #凡没有取任何数字，都为无情况
    dp[s][0]=''
for s in range(m+1): #凡位数为0也是无
    dp[0][s]=''
#作小偷背包
for i in range(1,n+1):
    for j in range(1,m+1):
        if weight[i-1]>j:#该整数位数大于现位数，不可取
            dp[i][j]=dp[i-1][j]
        else:
            dp[i][j]=str(max(c(dp[i-1][j]),int(arr[i-1]+dp[i-1][j-weight[i-1]])))
print(dp[n][m])

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20182014.png?raw=true)




### 02811: 熄灯问题

brute force, http://cs101.openjudge.cn/practice/02811

思路：

太难了，考试出这种题必死无疑，看题目其实一开始也没怎么看得懂，直到看了解答才哦，原来是要看第一行不同状态。。我还傻乎乎的就根据第一行现状态去推其他行，然后发现第一行全是0

gpt帮我生成了product的运行：

如下
```
按钮编号：1  2  3  4  5  6
可能状态：0  0  0  0  0  0  (不按)
         0  0  0  0  0  1  (按第6个)
         0  0  0  0  1  0  (按第5个)
          ...
         1  1  1  1  1  1  (全按)
```

代码：

```python
from copy import deepcopy
from itertools import product


TOGGLE = {0: 1, 1: 0}# 全局变量：翻转状态映射


def toggle_lights(matrix, i, j):
    matrix[i][j] = TOGGLE[matrix[i][j]]  # 当前灯
    matrix[i - 1][j] = TOGGLE[matrix[i - 1][j]]  # 上方灯
    matrix[i + 1][j] = TOGGLE[matrix[i + 1][j]]  # 下方灯
    matrix[i][j - 1] = TOGGLE[matrix[i][j - 1]]  # 左方灯
    matrix[i][j + 1] = TOGGLE[matrix[i][j + 1]]  # 右方灯


def solve_lights():

    matrix_backup = [[0] * 8] + [[0] + list(map(int, input().split())) + [0] for _ in range(5)] + [[0] * 8]

    # 遍历所有第1行触发方案（2^6种）
    for test_case in product(range(2), repeat=6):
        matrix = deepcopy(matrix_backup)  # 深拷贝初始矩阵
        triggers = [list(test_case)]  # 记录触发方案，第1行为当前遍历的方案

        # 遍历第2到第5行，根据上一行状态决定按下按钮
        for i in range(1, 6):
            for j in range(1, 7):
                if triggers[i - 1][j - 1]:
                    toggle_lights(matrix, i, j)

            triggers.append(matrix[i][1:7])# 记录当前行的触发结果（下一行的按钮按下状态）

        # 判断第6行是否全部熄灭
        if matrix[5][1:7] == [0] * 6:
            # 输出触发方案（不包括最后一行）
            for trigger in triggers[:-1]:
                print(" ".join(map(str, trigger)))
            return


solve_lights()

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20211643.png?raw=true)




### 08210: 河中跳房子

binary search, greedy, http://cs101.openjudge.cn/practice/08210/

思路：

先将最短跳跃距离设为路程的一半（理论上来说），接着算看要移除多少石头才能达此最短距离，如果移除多了石头，代表现最长的最短跳跃距离过长不可行，再缩短一半；如果还能移除石头，代表现非最长的最短跳跃距离

做了这题就顺便把类似的 04135:月度开销 自己做了，但还是有区别，月度开销是求最小值，选取left有难度

代码：

```python
def binary_stone(s):
    M=0
    s_now=0
    for i in range(1,n+2):
        if stone[i]-s_now<s:
            M+=1
        else:
            s_now=stone[i]

    if M>m:
        return True
    else:
        return False



L,n,m=map(int,input().split())
stone=[0]
for _ in range(n):
    stone.append(int(input()))
stone.append(L)

left=0
right=L+1
ans=0
while left<right:
    mid=(left+right)//2
    if binary_stone(mid):
        right=mid
    else:
        left=mid+1
        ans=mid
print(ans)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20215806.png?raw=true)




## 2. 学习总结和收获

做完了这次作业，有种上机要完蛋的状态，做搜索题目反而容易，这次倒是被各种bruteforce和greedy整得好惨，主要是这两种类型题目描述都很怪异，有时看了很久都读不懂题目要求，但是一看解答发现原理又很简单，这类题目又千奇百怪不知怎么能掌握

这周大概将cheatsheet整理好了，除了一些常用的语法、函数，也把一些题目放进去供参考，希望考试时能有所帮助，也希望上机真的不要那么难，因为我看了笔试发现更要完~

目前自己的题库收有169题，少于每日选做+作业的数量，只能选择性的去做每日选做了

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>