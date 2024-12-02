# Assignment #10: dp & bfs

Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### LuoguP1255 数楼梯

dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：

本题先用了bfs思考，一开始先套模板，发现只需要对pos（第pos个阶梯）处理就好，然后不知道该怎么记录路径数，就问了gpt，用ways进行储存，方法是如果你能走到 pos+1（即 pos+1 <= n），你就更新 ways[pos+1]，增加从 pos 到 pos+1 的路径数，对于pos+2一样，ways[n] 就会是从第0阶到达第 n 阶的所有不同路径数

本题其实用dp更快，状态方程是该阶梯路径数就等于n-1和n-2阶梯路径数和，得注意的是dp[0]和dp[1]=1

代码：

```python
from collections import deque

def bfs(n):
    q=deque([0])

    inq=set([0])

    ways=[0]*(n+1)
    ways[0]=1
    while q:
        pos=q.popleft()
        if pos + 1 <= n:
            if pos + 1 not in inq:
                inq.add(pos + 1)
                q.append(pos + 1)
            ways[pos + 1] += ways[pos]

        if pos+2<=n:
            if pos + 2 not in inq:
                inq.add(pos+2)
                q.append(pos+2)
            ways[pos+2]+=ways[pos]


    return ways[n]


n=int(input())
print(bfs(n))
```

```python

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

代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-26%20195146.png?raw=true)




### 27528: 跳台阶

dp, http://cs101.openjudge.cn/practice/27528/

思路：

做完第一题基本上2分钟内就搞定第二题，因为基本一样，改动的地方是不止从n-1和n-2抵达，还可以从n-j抵达，最大就是从n-n=0处一步到位

代码：

```python
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



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-26%20202130.png?raw=true)




### 474D. Flowers

dp, https://codeforces.com/problemset/problem/474/D

思路：

花了一小时多想不出，看解答又看了1小时多才理解，一开始也没明白MOD拿来干什么，dp一开始不知道怎么设，看了解答理解了好一阵子，其实就是把当前是红花还是黄花情况做个汇总，如果是红花那前面什么花都可以，如果是黄花则需要k-1个黄花成立


代码：

```python

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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-27%20214432.png?raw=true)




### LeetCode5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：

这题一开始就没想到要怎么用dp，反而是想起了二分查找用在回文序列非常好用，但是现在想如果要用dp可能就是i=left，j=right这样去改写即可；另外回文查找里容易忽略的就是中间也重复如abba这样的结构的，查找方式有别于aba

代码：

```python
def longest_palindrome(s):
    def find_palindrome(left,right):
        while left>=0 and right<len(s) and s[left]==s[right]:
            left-=1
            right+=1
        return left+1,right-1

    start,end=0,0
    for i in range(len(s)):
        l1,r1=find_palindrome(i,i)
        l2,r2=find_palindrome(i,i+1) #应对回文序列中间非单字情况如abba
        if r1-l1>end-start:
            start,end=l1,r1
        if r2-l2>end-start:
            start,end=l2,r2
    return s[start:end+1]

s=input()
print(longest_palindrome(s))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-27%20221357.png?raw=true)






### 12029: 水淹七军

bfs, dfs, http://cs101.openjudge.cn/practice/12029/

思路：

想着要用dfs解，已经写好了，题目测试数据和自己造的也过了，可是oj上就是一直re，看了答案就改一点还是re，最后改到跟答案99%接近了都，好麻烦啊啊，为什么输入那么麻烦，明明是一题很简单道的dfs或bfs题，我看了bfs答案也是一样麻烦

代码：

dfs

```python

import sys

sys.setrecursionlimit(300000)
input=sys.stdin.read


def dfs(x, y, water_height_value, m, n, h, water_height):
    move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx,dy in move:
        nx,ny=x+dx,y+dy
        if 0<=nx<m and 0<=ny<n and h[nx][ny]<water_height_value:
            if water_height[nx][ny]<water_height_value:
                water_height[x][y]=water_height_value
                dfs(nx, ny, water_height_value, m, n, h, water_height)

def main():
    data=input().split()
    idx=0
    k=int(data[idx])
    idx+=1
    results=[]
    for _ in range(k):
        m,n=map(int,data[idx:idx+2])
        idx+=2
        h=[]
        for _ in range(m):
            h.append(list(map(int,data[idx:idx+n])))
            idx+=n
        water_height=[[0]*n for _ in range(m)]
        i,j=map(int,data[idx:idx+2])
        i,j=i-1,j-1
        idx+=2
        p=int(data[idx])
        idx+=1
        for _ in range(p):
            a,b=map(int,data[idx:idx+2])
            idx+=2
            a,b=a-1,b-1
            if h[a][b]<=h[i][j]:
                continue
            dfs(a,b,h[a][b],m,n,h,water_height)

        results.append("Yes" if water_height[i][j]>0 else "No")
    sys.stdout.write("\n".join(results)+'\n')

if __name__=='__main__':
    main()
```

bfs（非自己尝试）
```python
from collections import deque
import sys
input = sys.stdin.read

# 判断坐标是否有效
def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n

# 广度优先搜索模拟水流
def bfs(start_x, start_y, start_height, m, n, h, water_height):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    q = deque([(start_x, start_y, start_height)])
    water_height[start_x][start_y] = start_height

    while q:
        x, y, height = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if is_valid(nx, ny, m, n) and h[nx][ny] < height:
                if water_height[nx][ny] < height:
                    water_height[nx][ny] = height
                    q.append((nx, ny, height))

# 主函数
def main():
    data = input().split()  # 快速读取所有输入数据
    idx = 0
    k = int(data[idx])
    idx += 1
    results = []

    for _ in range(k):
        m, n = map(int, data[idx:idx + 2])
        idx += 2
        h = []
        for i in range(m):
            h.append(list(map(int, data[idx:idx + n])))
            idx += n
        water_height = [[0] * n for _ in range(m)]

        i, j = map(int, data[idx:idx + 2])
        idx += 2
        i, j = i - 1, j - 1

        p = int(data[idx])
        idx += 1

        for _ in range(p):
            x, y = map(int, data[idx:idx + 2])
            idx += 2
            x, y = x - 1, y - 1
            if h[x][y] <= h[i][j]:
                continue
            bfs(x, y, h[x][y], m, n, h, water_height)

        results.append("Yes" if water_height[i][j] > 0 else "No")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    main()
```

代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-28%20000403.png?raw=true)




### 02802: 小游戏

bfs, http://cs101.openjudge.cn/practice/02802/

思路：

先说思路 就简单的上下左右走到目的地，取最短路径，麻烦的是不知怎么搞的命名测试数据能过但是就是RE/WA，只能照着参考答案修改，用了enumerate，我觉得关键就是在inq（或者visited）里要包括标记方向i，很大可能就是这个导致的我不通过


代码：

```python
from collections import deque

move = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def bfs(start, end, m, n, maze):
    global ans
    inq = set()

    q = deque()
    n_x, n_y = start
    q.append((n_x, n_y, -1, 0))  # 方向初始为-1，步数为0


    while q:
        x, y, now_dir, seg = q.popleft()
        if (x,y) == end:
            ans.append(seg)
            break
        for i,(dx, dy) in enumerate(move):
            nx, ny = x + dx, y + dy

            if 0 <= nx < m + 2 and 0 <= ny < n + 2 and ((nx, ny,i)) not in inq:
                new_dir = i
                if (nx, ny) == end:
                    if new_dir == now_dir:
                        ans.append(seg)
                        continue
                    else:
                        ans.append(seg + 1)
                        continue
                elif maze[nx][ny] != 'X':  # 不为障碍物
                    inq.add((nx, ny,i))
                    if new_dir != now_dir:
                        q.append((nx, ny, new_dir, seg + 1))
                    else:
                        q.append((nx, ny, new_dir, seg))

    if len(ans) == 0:
        return -1
    else:
        return min(ans)


board_n = 1

while True:
    w, h = map(int, input().split())
    if w == 0 and h == 0:
        break
    maze = [' ' * (w + 2)] + [' ' + input() + ' ' for _ in range(h)] + [' ' * (w + 2)]
    p_n = 1
    print(f"Board #{board_n}:")
    while True:
        ans = []
        y1, x1, y2, x2 = map(int, input().split())
        if x1 == y1 == y2 == x2 == 0:
            break
        start = (x1, y1)
        end = (x2, y2)
        seg = bfs(start, end, h, w, maze)

        if seg == -1:
            print(f"Pair {p_n}: impossible.")
        else:
            print(f"Pair {p_n}: {seg} segments.")
        p_n += 1

    print()
    board_n += 1

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-28%20165547.png?raw=true)




## 2. 学习总结和收获

总结，这次作业好难啊！bfs，dfs题目理解起来不难，套个模板可以开写，但是麻烦在于太长，要debug真的得看好半天，看解答也得对照好久。。。

难的地方是dp我自己还是不能写出来，这次作业大部分都得寻求参考答案或者ai帮忙，希望考试别那么难

我自己去做了23年秋的上机考，AC4道（其中一道是bfs所以看了模板，考试应该会把bfs抄在cheatsheet），另外两道是一题dp一题用permutation实在不会，希望考试不会那么难堪。。，目前也在cheatsheet写了语法的部分，接着就是写点算法和常需要的一些代码吧



<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>