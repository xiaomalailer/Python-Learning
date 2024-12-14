# Assignment #C: 五味杂陈 

Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 1115. 取石子游戏

dfs, https://www.acwing.com/problem/content/description/1117/

思路：

根据提示和逻辑，当a>=b时，先取该手的获胜，当然因为不考虑<1情况否则处理不了，所以就要在a<b时换位，然后当a//b<2时，胜负未分，进入下一次抓取，那么下一次抓取势必会让，a<b，所以要换位（b,a-b)，我认为第二种写法好理解，就按抓取步数，偶数为后手，奇数为先手者以作判定该轮谁在抓取


代码：

```python

def game(a,b):
    if (a//b)>=2 or a==b:
        return True
    else:
        return  not game(b,a-b)


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

        print('win' if game(a,b) else 'lose')
```

以下写法也行
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

代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20144710.png?raw=true)




### 25570: 洋葱

Matrices, http://cs101.openjudge.cn/practice/25570

思路：

观察可发现，每层拿取其实就是上面一层拿，下面一层拿 左右两边拿，就按照这样策略一层层进行计算即可


代码：

```python
n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

max_sum = 0
layer = 0

while layer < (n + 1) // 2:
    layer_sum = 0

    for i in range(layer, n - layer):# 上边界
        layer_sum += matrix[layer][i]

    if layer != n - layer - 1:#下边界
        for i in range(layer, n - layer):
            layer_sum += matrix[n - layer - 1][i]

    # 左右边界
    for i in range(layer + 1, n - layer - 1):
        layer_sum += matrix[i][layer]  # 左边
        layer_sum += matrix[i][n - layer - 1]  # 右边

    max_sum = max(max_sum, layer_sum)

    layer += 1

print(max_sum)

```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20201614.png?raw=true)




### 1526C1. Potions(Easy Version)

greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

思路：

完蛋！完全不认识heapq，问gpt从0学起，才了解堆的性质，会动态的处理堆顶保证它最小，所以此题是一旦发现喝了该药生命<0，那就不要喝or如果大于之前喝过的最小值，那就做替换（方便可能之后喝下更多）

代码：

```python
import heapq

def max_p(n,potion):
    health_now=0
    cnt=0
    heap=[]

    for p in potion:
        if health_now+p>=0:
            heapq.heappush(heap,p)#喝下当前药水
            health_now+=p
            cnt+=1
        elif heap and p>heap[0]:#喝下后生命值为负且大于先前喝下的最小值
            health_now+=p-heapq.heappop(heap)
            heapq.heappush(heap,p)
    return cnt

n=int(input())
potion=list(map(int,input().split()))
print(max_p(n,potion))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20220930.png?raw=true)




### 22067: 快速堆猪

辅助栈，http://cs101.openjudge.cn/practice/22067/

思路：

提示给了需要用辅助栈，鄙人不注意，即刻爆栈，辅助栈就是来维护最小值用的，之后不注意没有在pop时也把辅助栈pop也会造成错误，维护方法其实就是比较现在push的和之前栈顶（最小值），谁最小谁入栈（栈顶）


代码：

```python
pig=[]
hlp=[]
while True:
    try:
        s=input().split()
        if s[0]=='pop':
            if pig:
                pig.pop()
                if hlp:
                    hlp.pop()
        elif s[0]=='min':
            if hlp:
                print(hlp[-1])
        else:
            pig.append(int(s[1]))
            if not hlp:
                hlp.append(int(s[1]))
            else:
                x=hlp[-1]
                hlp.append(min(x,int(s[1])))

    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20230230.png?raw=true)




### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/

思路：

不信邪用bfs结果做不到，遂去了解dijkstra，原来是potion那题的heapq，这题比较形象点，主要是去走消耗小的，而用pq去维护堆顶的最小值，一种不同版本的bfs的感觉

代码：

```python
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20230230.png?raw=true)




### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129/

思路：

这题在月考前做过了，所以会写，不然其实这题一开始看到的时候是卡了很久看解答的，关键点在于temp=(time+1)%k上，就是判断石头会不会消失，如果你已经访问过 (temp, nx, ny) 这个状态，意味着你在相同的时间周期和相同的位置上已经做过了处理，后面步骤都是一样的，也就是说你在某点处时时间是t还是t+k是一样情况

代码：

```python
from collections import deque

move=[(1,0),(-1,0),(0,1),(0,-1)]
def bfs(start_x,start_y):
    q=deque()
    q.append((0,start_x,start_y))
    inq=set()
    inq.add((0,start_x,start_y))
    while q:
        time,x,y=q.popleft()
        temp=(time+1)%k
        for dx,dy in move:
            nx,ny=x+dx,y+dy
            if 0<=nx<r and 0<=ny<c and (temp,nx,ny) not in inq:
                if maze[nx][ny]=='E':
                    return time+1
                elif maze[nx][ny]!='#' or temp==0:
                    inq.add((temp,nx,ny))
                    q.append((time+1,nx,ny))
    return 'Oop!'

t=int(input())
for _ in range(t):
    r,c,k=map(int,input().split())
    maze=[list(input()) for _ in range(r)]
    for i in range(r):
        for j in range(c):
            if maze[i][j]=='S':
                print(bfs(i,j))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20222501.png?raw=true)




## 2. 学习总结和收获

完蛋了完蛋了，就算是这次作业的难度也要完蛋，首先就是potions和走山路那题用到堆，但是我压根没去学。。。理解了好久到底是怎幺维护堆顶的，感觉理解能力好差

感觉自己是一看题目就会很慌不知道怎么办，但是如果有提示就会有思路去做，0到1比1到100难好多。。。

还有两周时间，抓紧练习dp，考试时大概率会暴力解题了，希望题目不要那么严格

cheatsheet可以打印的吗？应该会将一些经典例题收录下来，不会时候去找灵感，还是希望自己能AC4以上，加油~~

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>