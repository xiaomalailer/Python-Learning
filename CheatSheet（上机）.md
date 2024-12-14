## 一、有关语法

### 输入
```n = int(input())```:输入 n为数字

```list1 = list(map(int, input().split()))``` ：输入一行形成列表 其中```input().split()```将整行输入按空格分割为多个部分 

```M, N = map(int, input().split())```一次读取两个数（经典必会）

```arr.append(tuple(map(int, input().split())))``` 形成list中的tuple

### 输出
```print("{a:._f}"))```：输出a保留_位小数

```print(*arr)```：列表中的所有元素展开为独立的参数，方便输出

```print(,end='')```： 就是输出后不要换下一行(也可用于输出后添加东西在后边）

```print(f"{}")```：可用于输出一些符号代表数值同时又不会有空格

```print(, sep='\n')```：输出每个元素占一行


### 函数（简单的不收录）

import math -> ```math.ceil()``` :向上取整 ;``` math.log() ```:可取logn

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

```from itertools import permutations``` ``` permutations(n)``` :全排列数字

```enumerate(perm)```将数组转化为（索引，元素）

``` a==a[::-1] ```判断回文方法之一

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

### 使用埃氏筛法生成所有小于等于 10^6 的质数（见于Tprimes）
```
def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False  # 0 和 1 不是质数
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return is_prime
```

### 矩阵相乘
```
for i in range(n):
    for j in range(n):
        for k in range(n):
            result[i][j] += arr_1[i][k] * arr_2[k][j]
```
### dfs

搜索题目通常包含一个地图：```maze = [list(map(int, input().split())) for _ in range(n)]```

因此一定有移动方向：```directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上```

为了避免重复经过：```visited = [[False] * m for _ in range(n)]  # 标记访问``` 这个视情况而定

因为dfs为深度搜索，搜完一条路需退回去再行其他路，所以需要回溯：```visited[nx][ny] = False``` 有时也会换成pop()

一般上按下例写法：

例题1：矩阵最大权值路径（包括记载路径）
```
# 读取输入
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

# 定义方向
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
visited = [[False] * m for _ in range(n)]  # 标记访问
max_path = []
max_sum = -float('inf')  # 最大权值初始化为负无穷

# 深度优先搜索
def dfs(x, y, current_path, current_sum):
    global max_path, max_sum

    # 到达终点，更新结果
    if (x, y) == (n - 1, m - 1):
        if current_sum > max_sum:
            max_sum = current_sum
            max_path = current_path[:]
        return

    # 遍历四个方向
    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        # 检查边界和是否访问过
        if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
            # 标记访问
            visited[nx][ny] = True
            current_path.append((nx, ny))

            # 递归搜索
            dfs(nx, ny, current_path, current_sum + maze[nx][ny])

            # 回溯
            current_path.pop()
            visited[nx][ny] = False

# 初始化起点
visited[0][0] = True
dfs(0, 0, [(0, 0)], maze[0][0])

# 输出结果
for x, y in max_path:
    print(x + 1, y + 1)
```

例2：螃蟹采蘑菇
```
# 蒋子轩，25572:螃蟹采蘑菇
def dfs(x,y):
    if matrix[x][y]==9:
        print('yes')
        exit()
    for k in range(4):
        nx,ny=x+dx[k],y+dy[k]
        if 0<=nx<n and 0<=ny<n and (nx,ny) not in visited and matrix[nx][ny]!=1:
            if type==1:#垂直移动
                if nx==n-1 or matrix[nx+1][ny]==1:
                    continue
                if matrix[nx+1][ny]==9:
                    print('yes')
                    exit()
            else:#水平移动
                if ny==n-1 or matrix[nx][ny+1]==1:
                    continue
                if matrix[nx][ny+1]==9:
                    print('yes')
                    exit()
            visited.add((nx,ny))
            dfs(nx,ny)
n=int(input())
dx=[1,0,0,-1]
dy=[0,1,-1,0]
matrix=[list(map(int,input().split())) for _ in range(n)]
visited=set()
for i in range(n):
    for j in range(n):
        if matrix[i][j]==5:
            if i<n-1 and matrix[i+1][j]==5:
                type=1 #起始垂直
            else:
                type=2 #起始水平
            dfs(i,j)
            print('no')
            exit()
```

例3：马走日
```
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
### bfs

首先必须要有```from collections import deque```

接着如果是地图形就要有移动方向

其他要有的代码参照下例

例题1：寻宝
```
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

例2：体育游戏跳房子（23年考题）
```
from collections import deque

def bfs(s,e):
    q=deque()
    q.append((0,s,''))
    inq=set()
    inq.add(s)

    while q:
        step,pos,path=q.popleft()
        if pos==e:
            return step,path
        if pos*3 not in inq:
            q.append((step+1,pos*3,path+'H'))
            inq.add(pos*3)
        if int(pos//2) not in inq:
            inq.add(int(pos//2))
            q.append((step+1,int(pos//2),path+'O'))



while True:
    n,m=map(int,input().split())
    if n==0 and m==0:
        break
    step,path=bfs(n,m)
    print(step)
    print(path)
```

例2：取石子游戏（其实是递归）

当a>=b时，先取该手的获胜，当然因为不考虑<1情况否则处理不了，所以就要在a<b时换位，然后当a//b<2时，胜负未分，进入下一次抓取，那么下一次抓取势必会让，a<b，所以要换位（b,a-b)，
```
def dfs(a, b, cnt):
    if (a // b >= 2 or a == b):  
        return cnt % 2 != 0  
    return dfs(b, a - b, cnt + 1)  


while True:
    a,b=map(int,input().split())
    if a==b==0:
        break

    if a>=b and (a//b)>=2 or a==b:
        print('win')
    else:
        if a<b:
            temp=a
            a=b
            b=temp

        print('win' if dfs(a,b,1) else 'lose')
```

例3：变换的迷宫
```
from collections import deque

move=[(-1,0),(1,0),(0,-1),(0,1)]

def bfs(start_x,start_y):
    q=deque()
    inq=set()
    inq.add((0,start_x,start_y))
    q.append((0,start_x,start_y))
    while q:
        time,x,y=q.popleft()
        for i in range(4):
            nx,ny=x+move[i][0],y+move[i][1]
            temp=(time+1)%K
            if 0<=nx<R and 0<=ny<C and (temp,nx,ny) not in inq:
                cur=maze[nx][ny]
                if cur=='E':
                    return time+1
                elif cur!='#' or temp==0:
                    q.append((time+1,nx,ny))
                    inq.add((temp,nx,ny))
    return 'Oop!'

T=int(input())
for _ in range(T):
    R,C,K=map(int,input().split())
    maze = [list(input()) for _ in range(R)]
    for i in range(R):
        for j in range(C):
            if maze[i][j] == 'S':
                print(bfs(i, j))
```

特例：dijkstra
```
# 输入处理
from heapq import heappop, heappush

move = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def dijkstra(start_x, start_y, end_x, end_y):
    pq = []  
    heappush(pq, (0, start_x, start_y))
    visited = set()  

    while pq:
        cost, x, y = heappop(pq)

        if x == end_x and y == end_y:
            return cost

        if (x, y) in visited:
            continue
        visited.add((x, y))

        
        for dx, dy in move:
            nx, ny = x + dx, y + dy

           
            if 0 <= nx < m and 0 <= ny < n and maze[nx][ny] != '#':
                new_cost = abs(int(maze[nx][ny]) - int(maze[x][y]))
                heappush(pq, (cost + new_cost, nx, ny))#确保cost最小在堆顶

    return 'NO'


# 输入处理
m, n, p = map(int, input().split())
maze = [input().split() for _ in range(m)]

for _ in range(p):
    x1, y1, x2, y2 = map(int, input().split())
    if maze[x1][y1] == '#' or maze[x2][y2] == '#':
        print('NO')
    else:
        print(dijkstra(x1, y1, x2, y2))

```
### dp动态规划

**dp动态规划解题五要素：**

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组
   
#### 0-1背包

例1：小偷背包
dp数组意义：只能放前 i个物品的情况下，容量为 j 的背包所能达到的最大总价值

假设当前已经处理好了前 `i-1` 个物品的所有状态，那么对于第 `i` 个物品，当其不放入背包时，背包的剩余容量不变，背包中物品的总价值也不变，故这种情况的最大价值为 $f_{i-1,j}$；当其放入背包时，背包的剩余容量会减小 $w_{i}$，背包中物品的总价值会增大 $v_{i}$，故这种情况的最大价值为 $f_{i-1,j-w_{i}}+v_{i}$
```
n,b=map(int, input().split())
price=[0]+[int(i) for i in input().split()]
weight=[0]+[int(i) for i in input().split()]
dp=[[0]*(b+1) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(1,b+1):
        if weight[i]<=j:
            dp[i][j]=max(price[i]+dp[i-1][j-weight[i]], dp[i-1][j])
        else:
            dp[i][j]=dp[i-1][j]
print(dp[-1][-1])
```

例2：最大整数 （23年考题）

dp[i][j]状态定义为在l的前i个数中随意选择，满足拼凑起来不超过j位，得到的整数的最大可能数值。那么dp[n][m]即为所求

```
def f(string):
    if string=='':
        return 0
    else:
        return int(string)

m=int(input())#最大位数
n=int(input())#正整数数量
l=input().split()

#冒泡排序(选择第一个数字）
for i in range(n):
    for j in range(n-1-i):
        if l[j] + l[j+1] > l[j+1] + l[j]:
            l[j],l[j+1] = l[j+1],l[j]

weight=[]#每个元素的位数

for num in l:
    weight.append(len(num))

#dp[i][j]在前i数中选择，不超过j位，最大可能数值
dp=[['']*(m+1) for _ in range(n+1)]

for k in range(m+1):
    dp[0][k]=''#无法组成整数
for q in range(n+1):
    dp[q][0]=''#无法组成整数

for i in range(1,n+1):
    for j in range(1,m+1):
        if weight[i-1]>j:#不能选第i个，因为会超位数
            dp[i][j]=dp[i-1][j]
        else:#可以选第i个也可以不选
                dp[i][j]=str(max(f(dp[i-1][j]),int(l[i-1]+dp[i-1][j-weight[i-1]])))
print(dp[n][m])
```

#### 跳楼梯情况

例1-1：可以跳一步也可以跳两步
```
def dp_way(n):
    if n==0:
        return 0
    if n==1:
        return 1
    dp=[0]*(n+1)
    dp[0]=1
    dp[1]=1
    for i in range(2,n+1):
        dp[i]=dp[i-1]+dp[i-2]
    return dp[n]

n=int(input())
print(dp_way(n))
```

例1-2：有代价的跳
```
N = int(input())
h = [0] + list(map(int, input().split()))

def min_cost(N, h):
    if N == 1:
        return 0  # 只有一个石头，不需要跳跃
    if N == 2:
        return abs(h[2] - h[1])  # 直接从石头1跳到石头2
    
    # 初始化前两个跳跃的代价
    prev2 = 0  # 到第一个石头的代价
    prev1 = abs(h[2] - h[1])  # 到第二个石头的代价
    
    # 动态规划迭代
    for i in range(3, N + 1):
        curr = min(prev1 + abs(h[i] - h[i - 1]), prev2 + abs(h[i] - h[i - 2]))
        prev2 = prev1
        prev1 = curr
    
    return prev1

# 输出最终结果
print(min_cost(N, h))
```
例2-1：不限步数跳
```
def dp_way(n):

    if n==1:
        return 1
    dp=[0]*(n+1)
    dp[0]=1
    dp[1]=1
    for i in range(2,n+1):
        for j in range(1,i+1):
            dp[i]+=dp[i-j]
    return dp[n]

n=int(input())
print(dp_way(n))
```
例2-2：有代价
```
N, K = map(int, input().split())  # 输入 N 和 K
h = [0] + list(map(int, input().split()))  # 楼层高度数组，注意加一个 0 在前面方便索引

dp = [float('inf')] * (N + 1)  # 初始化 dp 数组，设置为极大值
dp[1] = 0  # 第一层跳跃代价为 0

# 计算从第 1 层开始到其他各层的最小代价
for i in range(2, N + 1):
    for j in range(1, min(K, i - 1) + 1):  # 从前面 K 层跳跃
        dp[i] = min(dp[i], dp[i - j] + abs(h[i] - h[i - j]))

print(dp[N])  # 输出最小代价
```
#### 每日任务安排 vacation
此种题目重点在于不能连续两天做一样的事情
例1：
```
n=int(input())
a=list(map(int,input().split()))
dp=[0]*(n)
if a[0]==0:
    dp[0]=1

for i in range(1,len(a)):
    if a[i]==0:
        dp[i]=1
        i+=1
        continue
    if a[i]==a[i-1] and a[i-1]!=3:
        a[i]=0
        dp[i]=1
        i+=1
        continue
    if a[i]==3 and a[i-1]!=3:
        a[i]=3-a[i-1]
    i+=1
print(sum(dp))
```

例2：
```
# 输入天数
N = int(input())

# 检查输入数据完整性
day = []
for _ in range(N):
    values = list(map(int, input().split()))
    day.append(values)


dp = [[0] * 3 for _ in range(N)]

# 第一天天收益
for i in range(3):
    dp[0][i] = day[0][i]

# 动态规划计算
for i in range(1, N):
    dp[i][0] = day[i][0] + max(dp[i-1][1], dp[i-1][2])
    dp[i][1] = day[i][1] + max(dp[i-1][0], dp[i-1][2])
    dp[i][2] = day[i][2] + max(dp[i-1][0], dp[i-1][1])

# 输出最大收益
print(max(dp[N-1][0], dp[N-1][1], dp[N-1][2]))
```

#### 最长连续子序列
```
s = input()
t = input()

# 动态规划求解 LCS
m, n = len(s), len(t)
dp = [[0] * (n + 1) for _ in range(m + 1)]

# 填充 dp 表
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s[i - 1] == t[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

# 回溯获得 LCS
i, j = m, n
ans = []
while i > 0 and j > 0:
    if s[i - 1] == t[j - 1]:
        ans.append(s[i - 1])
        i -= 1
        j -= 1
    elif dp[i - 1][j] > dp[i][j - 1]:
        i -= 1
    else:
        j -= 1

# 输出结果
print("".join(reversed(ans)))

```
#### 最长路径(拓扑排序）
例：
```
from collections import deque

# 输入节点数和边数
N, M = map(int, input().split())

# 邻接表表示图
adj = [[] for _ in range(N + 1)]
indegree = [0] * (N + 1)

# 输入边的信息
for _ in range(M):
    x, y = map(int, input().split())
    adj[x].append(y)
    indegree[y] += 1

# 拓扑排序
topological_order = []
queue = deque()

# 初始时，将所有入度为0的节点加入队列
for i in range(1, N + 1):
    if indegree[i] == 0:
        queue.append(i)

while queue:
    u = queue.popleft()
    topological_order.append(u)
    for v in adj[u]:
        indegree[v] -= 1
        if indegree[v] == 0:
            queue.append(v)

# 动态规划数组，用来存储每个节点的最长路径长度
dp = [0] * (N + 1)

# 根据拓扑排序进行DP更新
for u in topological_order:
    for v in adj[u]:
        dp[v] = max(dp[v], dp[u] + 1)

# 输出最长的路径长度
print(max(dp))
```

#### 用dp走迷宫情况
```
MOD = 10**9 + 7

# 输入处理
H, W = map(int, input().split())
maze = [input() for _ in range(H)]

# 初始化 DP 数组，dp[i][j] 表示到达 (i, j) 的路径数
dp = [[0] * W for _ in range(H)]
dp[0][0] = 1  # 起点路径数为 1

# 填充 DP 数组
for i in range(H):
    for j in range(W):
        if maze[i][j] == '#':  # 如果是墙壁，无法通行
            continue
        if i > 0:  # 从上方走到当前位置
            dp[i][j] += dp[i - 1][j]
            dp[i][j] %= MOD
        if j > 0:  # 从左侧走到当前位置
            dp[i][j] += dp[i][j - 1]
            dp[i][j] %= MOD

# 输出到达右下角的路径数
print(dp[H - 1][W - 1])
```
#### 丢硬币
```
N = int(input()) 
p = list(map(float, input().split()))

# dp[i][j] 表示前 i 枚硬币中，恰好有 j 枚正面朝上的概率
dp = [[0] * (N + 1) for _ in range(N + 1)]

dp[0][0] = 1 # 初始状态：没有投掷硬币时，正面朝上的概率为 1

# 动态规划填表
for i in range(1, N + 1):  # 遍历第 i 枚硬币
    for j in range(1, i + 1):  # 遍历正面朝上的硬币数 j
        # 当前硬币正面朝上或反面朝上的情况
        dp[i][j] = dp[i - 1][j] * (1 - p[i - 1]) + dp[i - 1][j - 1] * p[i - 1]
    
    # 特殊情况：当前硬币正面朝上的次数为 0
    dp[i][0] = dp[i - 1][0] * (1 - p[i - 1])

# 计算至少有一半硬币正面朝上的概率
# 事件发生次数大于 N // 2 的情况，即 j > N // 2
result = sum(dp[N][j] for j in range(N // 2 + 1, N + 1))

# 输出结果
print(result)

```

#### 双dp情况

两个dp数组分别对应两种情况：

例1：土豪购物

设立两个数组，一个为没有放回物品，一个放回，dp1递推公式dp1[i]=max(dp1[i-1]+buy[i],buy[i])即选择这个物品是连续选择或单独选择，dp2增多考虑一个dp1[i-1]即放回物品（不选择当前物品）
```
buy=list(map(int,input().split(',')))
dp1=[0]*len(buy)
dp2=[0]*len(buy)
dp1[0]=buy[0]
dp2[0]=buy[0]
for i in range(1,len(buy)):
    dp1[i]=max(dp1[i-1]+buy[i],buy[i])
    dp2[i]=max(dp1[i-1],dp2[i-1]+buy[i],buy[i])
print(max(dp2))
```

例2：红蓝玫瑰

两个一维dp，一个是前n朵玫瑰全变红，记为Rn，一个是前n朵玫瑰全变蓝，记为Bn。 如果n+1朵玫瑰是红色，R(n+1)=Rn,B(n+1)可以通过魔法一由前n朵全是蓝色的玫瑰变来，也可以通过魔法二由前n朵全是红色的玫瑰变来。所以B(n+1)=min(Rn,Bn)+1。 如果n+1朵玫瑰是蓝色就反过来
```
flw=list(input())
r=[0]*(len(flw))
b=[0]*(len(flw))
if flw[0]=='R':
    r[0]=0
    b[0]=1
else:
    r[0]=1
    b[0]=0

for i in range(len(flw)-1):
    if flw[i+1]=='R':
        r[i+1]=r[i]
        b[i+1]=min(b[i],r[i])+1
    else:
        r[i+1]=min(r[i],b[i])+1
        b[i+1]=b[i]

print(r[-1])
```

例3：Basketball Exercise

篮球选人，有两行人员，所以对于第一行的可以不选，或者选，选则是现高度+在第二行选的高度（不能连续选同一行）
```
n=int(input())
h1=list(map(int,input().split()))
h2=list(map(int,input().split()))
dp1=[0]*n
dp2=[0]*n
dp1[0]=h1[0]
dp2[0]=h2[0]

for i in range(1,n):
    dp1[i]=max(dp1[i-1],dp2[i-1]+h1[i])
    dp2[i]=max(dp2[i-1],dp1[i-1]+h2[i])

print(max(dp1[n-1],dp2[n-1]))
```
#### 放苹果

如果苹果数多于盘子数，那么两种处理，

第一种：空一个盘，把苹果放在其他盘

第二种：每个盘子先放一颗苹果，然后再处理剩下的苹果
```
t=int(input())
for _ in range(t):
    m,n=map(int,input().split())
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][1]=1
    for i in range(n+1):
        dp[0][i]=1

    for i in range(1,m+1):
        for j in range(2,n+1):
            if j>i:
                dp[i][j]=dp[i][i]
            else:
                dp[i][j]=dp[i][j-1]+dp[i-j][j]
    print(dp[m][n])
```

#### flowers(类似红蓝玫瑰）
```
t,k=map(int,input().split())
MAX=1000000007
MOD=int(1e9+7)
MAXN=int(1e5+1)
dp=[0]*MAXN
s=[0]*MAXN
dp[0]=1
s[0]=1
for i in range(1,MAXN):
    if i>=k:
        dp[i]=(dp[i-1]+dp[i-k])%MOD
    else:
        dp[i]=dp[i-1]%MOD
    s[i]=(s[i-1]+dp[i])%MOD

for _ in range(t):
    a,b=map(int,input().split())
    print((s[b] - s[a - 1] + MOD) % MOD)
```

### 递归
**递归三法则**
1. 递归算法必须有一个**基准情形（递的终止，归的开始）**

2. 递归算法必须改变其状态并朝着基准情形前进。

3. 递归算法必须**调用自身**，即进行递归调用。

例：Converting an Integer to a String in Any Base
```
def to_str(n, base):
    # 定义用于转换的字符序列
    convert_string = "0123456789ABCDEF"

    # 基准情形：如果 n 小于基数，则直接返回对应的字符
    if n < base:
        return convert_string[n]
    else:
        # 递归调用：先处理商，再处理余数
        # 通过延迟连接操作，确保结果的顺序是正确的
        return to_str(n // base, base) + convert_string[n % base]


# 示例
print(to_str(10, 2))  # 输出: "1010"
print(to_str(255, 16))  # 输出: "FF"
```

例：汉诺塔问题
```
# https://blog.csdn.net/geekwangminli/article/details/7981570

# 将编号为numdisk的盘子从init杆移至desti杆 
def moveOne(numDisk : int, init : str, desti : str):
    print("{}:{}->{}".format(numDisk, init, desti))

#将numDisks个盘子从init杆借助temp杆移至desti杆
def move(numDisks : int, init : str, temp : str, desti : str):
    if numDisks == 1:
        moveOne(1, init, desti)
    else: 
        # 首先将上面的（numDisk-1）个盘子从init杆借助desti杆移至temp杆
        move(numDisks-1, init, desti, temp) 
        
        # 然后将编号为numDisks的盘子从init杆移至desti杆
        moveOne(numDisks, init, desti)
        
        # 最后将上面的（numDisks-1）个盘子从temp杆借助init杆移至desti杆 
        move(numDisks-1, temp, init, desti)

n, a, b, c = input().split()
move(int(n), a, b, c)
```

```
#简洁写法
def move(n,init:str,temp:str,des:str):
    if n==0:
        return
    move(n-1,init,des,temp)
    print(f"{n}:{init}->{des}")
    move(n-1,temp,init,des)

n,a,b,c=input().split()
move(int(n),a,b,c)
```

例：全排列
```

def work_p(idx, n, used, temp, result):
    # 如果 idx 超出 n，表示当前排列完成，将其添加到结果集中
    if idx == n + 1:
        result.append(temp[:])  # 将 temp 的副本添加到 result 中
        return  # 结束当前递归调用
    
    # 尝试将数字 1 到 n 放入当前排列的第 idx 个位置
    for i in range(1, n + 1):
        # 如果数字 i 未被使用，则将其添加到当前排列中
        if not used[i]:
            temp.append(i)       # 将数字 i 添加到当前排列 temp 中
            used[i] = True       # 标记数字 i 已被使用
            # 递归调用，继续填充下一个位置
            work_p(idx + 1, n, used, temp, result)
            # 回溯过程，将 i 从当前排列中移除并标记为未使用
            used[i] = False      # 撤销使用 i 的标记
            temp.pop()           # 移除最后添加的 i

def permutations(n):
    # 初始化结果列表，用于存储所有排列
    result = []
    # 初始化使用列表，记录每个数字是否已被使用
    used = [False] * (n + 1)  # 下标 0 不使用，因此长度为 n+1
    # 调用递归函数，从第一个位置开始填充排列
    work_p(1, n, used, [], result)
    
    # 输出结果中的每一个排列
    for ans in result:
        print(' '.join(map(str, ans)))  # 将排列中的数字转成字符串并用空格连接

# 从用户输入中读取 n 的值
n = int(input("请输入一个整数 n: "))
# 生成并打印从 1 到 n 的所有排列
permutations(n)
```
### 排队

```
N, D = map(int, input().split())  # 读取两个整数 N（学生数量）和 D（相邻身高差的最大值）
height = [0] * N  # 初始化一个大小为 N 的数组来存储学生的身高
check = [False] * N  # 初始化一个大小为 N 的布尔数组，记录每个学生是否已被处理

# 循环读取每个学生的身高
for i in range(N):
    height[i] = int(input())
    height_new = []  # 用于存储排序后的身高
while False in check:  # 只要还有未处理的学生，就继续执行
    i, l = 0, len(height)  # i 是当前学生的索引，l 是学生总数
    buffer = []  # 暂存当前可以排序的子序列
    while i < l:
        if check[i]:  # 如果当前学生已经处理，跳过
            i += 1
            continue
        if len(buffer) == 0:  # 如果缓冲区为空，开始处理一个新的子序列
            buffer.append(height[i])  # 把第一个未处理的学生加入 buffer
            maxh = height[i]  # 记录当前子序列的最大值
            minh = height[i]  # 记录当前子序列的最小值
            check[i] = True  # 标记该学生已处理
            continue

        # 如果当前学生的身高和子序列中的最大值、最小值的差都小于等于 D，则加入子序列
        maxh = max(height[i], maxh)
        minh = min(height[i], minh)
        if maxh - height[i] <= D and height[i] - minh <= D:
            buffer.append(height[i])
            check[i] = True  # 标记该学生已处理
        i += 1  # 处理下一个学生
        
    buffer.sort()  # 对找到的子序列进行排序，确保字典序最小
    height_new.extend(buffer)  # 将排序好的子序列加入最终的结果中

print(*height_new, sep='\n')  # 按顺序输出新的身高排列，每个身高占一行
```

#### 排队做实验

第一位的实验时长是后边每个人都要等的，*(n-1),以此类推
```
n=int(input())
ptime=list(map(int,input().split()))
result=[(ptime[i],i+1) for i in range(n)]
result=sorted(result)
for i in range(0,n):
    print(result[i][1],end=' ')
sum_t=0
cnt=n-1
for i in range(0,n-1):
    sum_t+=cnt*result[i][0]
    cnt-=1
print()
print(f"{sum_t/n:.2f}")
```
#### 病人排队

```
n=int(input())
old=[]
yng=[]

for i in range(n):
    id,age=input().split()
    if int(age)>=60:
        age=int(age)
        old.append((age,id,i))
    else:
        yng.append(id)
old=sorted(old,key=lambda x:(-x[0],x[2]))

for i in range(len(old)):
    print(old[i][1])
for i in yng:
    print(i)
```

### 其他经典题
#### 垃圾炸弹
```
d=int(input())
n=int(input())
mos=[[0]*1025 for _ in range(1025)]
for _ in range(n):
    a,b,c=map(int,input().split())# 读取屏幕的坐标 (a, b) 和清扫垃圾数c
    for i in range(max(a-d,0),min(a+d+1,1025)):
        for j in range(max(b-d,0),min(b+d+1,1025)):
            mos[i][j]+=c
#对于每一个炸弹，计算其影响范围，即在 (a-d, a+d) 和 (b-d, b+d) 范围内的所有点（这是一个矩形区域），这个范围内的每个点都会增加c

res=max_point=0
for i in range(1025):
    for j in range(1025):
        if mos[i][j]>max_point:
            max_point=mos[i][j]#当前点的c大于 max_point，更新 max_point 并将 res 重置为 1。
            res=1
        elif mos[i][j]==max_point:#当前点的c值等于 max_point，增加 res 的值，表示该点也有最大c
            res+=1
    
print(res,max_point)
```

#### 罗马数字与整数交换
```
def roman_to_int(roman):
    rome_1 = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    special_cases = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    
    i, total = 0, 0
    while i < len(roman):
        if i < len(roman) - 1 and roman[i:i+2] in special_cases:  # 处理特殊情况（如IV, IX等）
            total += special_cases[roman[i:i+2]]
            i += 2
        else:
            total += rome_1[roman[i]]
            i += 1
    return total

def int_to_roman(num):
    val_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), 
               (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), 
               (5, 'V'), (4, 'IV'), (1, 'I')]
    
    roman = []
    for val, symbol in val_map:
        while num >= val:
            roman.append(symbol)
            num -= val
    return ''.join(roman)

arr = input()

if arr.isdigit():  # 如果输入是数字，转换为罗马数字
    num = int(arr)
    print(int_to_roman(num))
else:  # 如果输入是罗马数字，转换为整数
    print(roman_to_int(arr))
```

#### 日期加法
```
def days_ofmonth(year,month):
    if month in [1,3,5,7,8,10,12]:
        return 31
    elif month in [4,6,9,11]:
        return 30
    elif month ==2:
        return 29 if is_leapyear(year) else 28

def is_leapyear(year):
    return(year%4==0 and year%100!=0) or (year%400==0)

date_1=input()
add_day=int(input())

year=int(date_1[0:4])
month=int(date_1[5:7])
day=int(date_1[8:])

day+=add_day
while day>days_ofmonth(year,month):
    day-=days_ofmonth(year,month)
    month+=1
    if month>12:
        month=1
        year+=1
print(f'{year}-{month:02d}-{day:02d}')
```

#### 装箱问题
```
import math

rest = [0,5,3,1]  # 当装了3x3的物品后，每个箱子还能装多少2x2的物品

while True:
    # 输入六种物品数量
    a, b, c, d, e, f = map(int, input().split())

    # 如果所有输入都为0，停止程序
    if a + b + c + d + e + f == 0:
        break

    # 每个4x4, 5x5, 6x6都需要单独的箱子，直接累加
    boxes = d + e + f

    # 计算3x3物品需要的箱子数量，4个3x3可以填满一个箱子
    boxes += math.ceil(c/4)

    # 计算能和4x4及3x3一起装的2x2数量
    spaceforb = 5*d + rest[c%4]

    # 如果2x2物品超出了可装空间，增加所需的箱子数量
    if b > spaceforb:
        boxes += math.ceil((b - spaceforb)/9)

    # 计算剩余的1x1物品可以放的空间
    spacefora = boxes*36 - (36*f + 25*e + 16*d + 9*c + 4*b)

    # 如果1x1物品超出了现有箱子的空间，增加所需的箱子数量
    if a > spacefora:
        boxes += math.ceil((a - spacefora)/36)

    # 输出所需的箱子数量
    print(boxes)

```
