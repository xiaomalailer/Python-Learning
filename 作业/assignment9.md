# Assignment #9: dfs, bfs, & dp

Updated 2107 GMT+8 Nov 19, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 18160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/practice/18160

思路：

简直就是套模板题，只不过多了左上左下右上右下要去考虑
耗时约20分钟（debug。。）

代码：

```python
move=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]#上下左右左上右上左下右下

area=0
def dfs(x, y):
    global area
    if x < 0 or x >= len(maze) or y < 0 or y >= len(maze[0]) or maze[x][y] == '.':
        return
    maze[x][y] = '.'
    area += 1
    for dx, dy in move:
        dfs(x + dx, y + dy)


t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    maze=[]
    maze.append(['.']*(m+2))
    for _ in range(n):
        maze.append(['.']+list(input())+['.'])
    maze.append(['.']*(m+2))

    result=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            if maze[i][j]=='W':
                area=0
                dfs(i,j)
                result=max(result,area)

    print(result)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-20%20200127.png?raw=true)




### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

思路：

纯属照搬了bfs模板即可，与sy318数字操作那题非常类似


代码：

```python
from collections import deque

move=[(-1,0),(1,0),(0,1),(0,-1)]

def bfs(m,n,maze):
    inq=set()
    start=(0,0)
    inq.add(start)
    q=deque()
    q.append([start,0])

    while q:
        (x,y),step=q.popleft()
        if maze[x][y]==1:
            return step

        for dx,dy in move:
            nx,ny=x+dx,y+dy
            if 0<=nx<m and 0<=ny<n and (nx,ny) not in inq and maze[nx][ny]!=2:
                inq.add((nx,ny))
                q.append([(nx,ny),step+1])

    return "NO"




m,n=map(int,input().split())
maze=[list(map(int,input().split())) for _ in range(m)]
print(bfs(m,n,maze))
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-20%20210846.png?raw=true)




### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123

思路：

马走日，所以把移动方式改下，这题需要进行回溯，所以有pop


代码：

```python
# 马走日的移动方式
move = [(2, 1), (2, -1), (-2, 1), (-2, -1), (-1, 2), (-1, -2), (1, -2), (1, 2)]

path_sum=0

def dfs(x, y, n, m, path):
    global path_sum
    path.append((x, y))
    if len(path) == n * m:
        path_sum += 1
        path.pop()
        return

    for dx, dy in move:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in path:
            dfs(nx, ny, n, m, path)

    path.pop()


t = int(input())
for _ in range(t):
    n, m, start_x, start_y = map(int, input().split())
    path_sum = 0
    dfs(start_x, start_y, n, m, [])
    print(path_sum)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-20%20214454.png?raw=true)




### sy316: 矩阵最大权值路径

dfs, https://sunnywhy.com/sfbj/8/1/316

思路：

因为之前看老师pdf时看过，就大概背着写了，主要是先像一般迷宫一样上下左右的试看，并计算权值和，期间记录途径，如果是最大权值，就记录此途径


代码：

```python
n,m=map(int,input().split())
maze=[list(map(int,input().split())) for _ in range(n)]

directions=[(0,1),(1,0),(-1,0),(0,-1)] #→ ↓ ← ↑
node=[[False]*m for _ in range(n)]
max_path=[]
max_sum=-float('inf')

def dfs(x,y,current_path,current_sum):
    global max_path,max_sum
    if (x,y)==(n-1,m-1):
        if current_sum>max_sum:
            max_sum=current_sum
            max_path=current_path[:]
        return

    for dx,dy in directions:
        nx,ny=x+dx,y+dy
        if 0<=nx<n and 0<=ny<m and not node[nx][ny]:
            node[nx][ny]=True
            current_path.append((nx,ny))
            dfs(nx,ny,current_path,current_sum+maze[nx][ny])

            current_path.pop()#回溯标志
            node[nx][ny]=False

node[0][0]=True
dfs(0,0,[(0,0)],maze[0][0])

for x,y in max_path:
    print(x+1,y+1)


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-20%20011057.png?raw=true)






### LeetCode62.不同路径

dp, https://leetcode.cn/problems/unique-paths/

思路：

动笔画图，即可得状态方程为上一格与左一格之和

代码：

```python
class Solution(object):
    def uniquePaths(self, m, n):
        dp=[[1]*n]+[[1]+[0]*(n-1) for _ in range(m-1)]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j]=dp[i-1][j]+dp[i][j-1]
        return(dp[-1][-1])

```

后看解答发现个蛮好的解法，是一行一行线性处理，即改状态为自己（其实就是上一个格子）+左边格子

```python
class Solution(object):
    def uniquePaths(self, m, n):
        crt=[1]*n
        for i in range(1,m):
            for j in range(1,n):
                crt[j]+=crt[j-1]
        return crt[-1]

```

代码运行截图 ==（至少包含有"Accepted"）==

![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-19%20182951.png?raw=true)




### sy358: 受到祝福的平方

dfs, dp, https://sunnywhy.com/sfbj/8/3/539

思路：

先制造一个10**9内的完全平方数，从最高位开始检查是否是完全平方数不是就纳入下一位数检查，是就检查下一位数是否是完全平方数（重复）

代码：

```python

def zhu_fu(A):
    square=set()
    i=1
    while i*i <10**9:
        square.add(i*i)
        i+=1

    digits=list(map(int,str(A)))
    def dfs(cnt):
        if cnt==len(digits):
            return True
        num=0
        for i in range(cnt,len(digits)):
            num=num*10+digits[i]
            if num in square:
                if(dfs(i+1)):
                    return True
        return False
    return "Yes" if dfs(0) else "No"

A=int(input())
print(zhu_fu(A))


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-20%20004447.png?raw=true)




## 2. 学习总结和收获

这次的作业表明看起来有难度，但是因为这个星期有去上课，学会了dfs和bfs的模板，套起来就很好用，不过就是对模板可能还不够熟悉，多做几道题就🆗了，可能之后会写在cheatsheet上方便考试用

对比起来，还是觉得dp好难，dfs和bfs倒是没想象中的难，不过现在应该掌握所有应学会的算法后，会开始刷题，期许自己能在期末有个好成绩

这次作业有几题喂了ai来优化，发现自己写的好多东西都太繁琐，有些压根没有，比如有些题目会要先输入测试数据数量，这个基本上可以简化成 for _ in range(int(input())

目前其实还没分清楚bfs和dfs对应的情景区别，应该还要多做几道题分清楚，不然考试时要都尝试感觉会耗时，比较不熟的是要输出路径那种，前几周摆烂，只是完成作业没去听课，这周有去听才发现听老师讲课其实蛮有用处的，比起自己摸索更快地能理

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>