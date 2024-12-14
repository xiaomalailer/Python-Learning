# Assignment #B: Dec Mock Exam大雪前一天

Updated 1649 GMT+8 Dec 5, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）⽉考： <mark> AC3 </mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E22548: 机智的股民老张

http://cs101.openjudge.cn/practice/22548/

思路：

AC, 试了4、5次不是TLE，就是memory问题，看来是不能接受O(N^2)，也就是说本来用dp做做不到，过后冷静下来简单解决

方法是将数组分成price价格和profit收益，price低则换，profit就是当天price-之前的min_price(其实就是当天卖出价格减去之前低价买入价）


代码：

```python
a=list(map(int,input().split()))
if not a:
    print(0)
else:
    min_price=float('inf')
    max_profit=0
for price in a:
    min_price=min(min_price,price)
    max_profit=max(max_profit,price-min_price)

print(max_profit)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-05%20180616.png?raw=true)




### M28701: 炸鸡排

greedy, http://cs101.openjudge.cn/practice/28701/

思路：

没有ac，看答案才会做，这题代码看着短但是思路可不太好想，我理解的是，先算个平均炸的时间，如果最大耗时鸡排比这个平均值大，那么就放着一直炸它，剩下的位置炸比较小的鸡排


代码：

```python
def max_fry_time(n,k,t):
    t.sort()
    sum_t=sum(t)
    while True:
        if (t[-1]>sum_t/k):
            k-=1
            sum_t-=t.pop()
        else:
            return sum_t/k

n,k=map(int,input().split())
t=list(map(int,input().split()))
result=max_fry_time(n,k,t)
print(f"{result:.3f}")
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-06%20180559.png?raw=true)




### M20744: 土豪购物

dp, http://cs101.openjudge.cn/practice/20744/

思路：

啊好后悔没仔细做这题，因为玫瑰、basketball都做过了，结果却没想到

设立两个数组，一个为没有放回物品，一个放回，dp1递推公式dp1[i]=max(dp1[i-1]+buy[i],buy[i])即选择这个物品是连续选择或单独选择，dp2增多考虑一个dp1[i-1]即放回物品（不选择当前物品）


代码：

```python
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-05%20193806.png?raw=true)




### T25561: 2022决战双十一

brute force, dfs, http://cs101.openjudge.cn/practice/25561/

思路：

没有AC，然后看答案了，但是有自己跟着写了一遍，要我写我写不出，但是还是理解了，就是各个情况下各别结算在各个店铺买了多少钱，在带进去算总共折扣价格，然后进入下个情况，比如第n个商品换成另外个店铺买，所以需要回溯，月考时读了下题就直接确定是我写不出的就放弃了，感觉这题与传统走迷宫的dfs很不像（不然就好像可以斜着走了哈哈）

代码：

```python
result=float('inf')
n,m=map(int,input().split())
store_prices=[input().split() for _ in range(n)]
coupons=[input().split() for _ in range(m)]

def dfs(store_prices,coupons,items=0,total_price=0,each_store_price=[0]*m):
    global result
    if items==n:
        coupon_price=0
        for i in range(m):
            store_p=0
            for coupon in coupons[i]:
                a,b=map(int,coupon.split('-'))
                if(each_store_price[i]>=a):
                    store_p=max(store_p,b)
            coupon_price+=store_p
        result=min(result,total_price-(total_price//300)*50-coupon_price)
        return

    for i in store_prices[items]:
        idx,p=map(int,i.split(':'))
        each_store_price[idx-1]+=p
        dfs(store_prices,coupons,items+1,total_price+p,each_store_price)
        each_store_price[idx-1]-=p
dfs(store_prices,coupons)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-06%20191633.png?raw=true)




### T20741: 两座孤岛最短距离

dfs, bfs, http://cs101.openjudge.cn/practice/20741/

思路：

AC，起初我是将函数short_bridge放在主函数下方的，但是一直TLE，直到移出来处理才AC，这点其实我也不解

先说下思路：这题我不是通过bfs找最短途径，因为最短途径其实就是两岛最短距离abs(x1 - x2) + abs(y1 - y2) - 1，是通过bfs找出第一个岛屿到底包含哪些点，第二个岛屿又有哪些点，明白这点后就简单多了（题解貌似不是这样想的？）

代码：

```python
from collections import deque

move=[(-1,0),(1,0),(0,-1),(0,1)]

def bfs(maze,visited,start_x,start_y,island_p):
    m=len(maze)

    q=deque([(start_x,start_y)])
    visited[start_x][start_y]=True
    while q:
        x,y=q.popleft()
        island_p.append((x,y))
        for dx,dy in move:
            nx,ny=x+dx,y+dy
            if 0<=nx<m and 0<=ny<m and not visited[nx][ny] and maze[nx][ny]==1:
                visited[nx][ny]=True
                q.append((nx,ny))

def short_bridge(maze):
    m = len(maze)
    visited = [[False] * n for _ in range(n)]
    is1, is2 = [], []
    found = False
    for i in range(n):
        for j in range(n):
            if maze[i][j] == 1 and not visited[i][j]:
                if not found:
                    bfs(maze, visited, i, j, is1)
                    found = True
                else:
                    bfs(maze, visited, i, j, is2)
                    break
    min_dis = float('inf')
    for x1, y1 in is1:
        for x2, y2 in is2:
            dis = abs(x1 - x2) + abs(y1 - y2) - 1
            min_dis = min(min_dis, dis) #求最短距离
    return min_dis

n=int(input())
maze=[list(map(int,input().strip())) for _ in range(n)]

print(short_bridge(maze))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-05%20183323.png?raw=true)




### T28776: 国王游戏

greedy, http://cs101.openjudge.cn/practice/28776zu

思路：

AC，大臣中，左右手相乘越大排后面（右手大排后面一般可以除以更大的数得到更小的金额，左手大也尽量排后面，以便左手相乘数小），然后根据此排序找出最大奖赏金额，这题大概是这次月考最简单的题了。。。

代码：

```python
n=int(input())
king=list(map(int,input().split()))
left_v=king[0]
right_v=king[1]
a=[]
for i in range(1,n+1):
    ple=list(map(int,input().split()))
    a.append((ple[0],ple[1]))
a.sort(key=lambda x:x[0]*x[1])
ans=0
for i in range(n):
    if(left_v//a[i][1]>ans):
        ans=left_v//a[i][1]
    left_v*=a[i][0]
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-05%20185839.png?raw=true)




## 2. 学习总结和收获

这次只AC了3呜呜，然后月考时有去翻了自己以前写过的题参考bfs，所以才做得出孤岛那题，第一题成功AC时时间都要过半，写bfs第五题花了很多时间，虽然感觉自己要做到，但是一直要debug，又不想放弃写了那么长的代码，可恶。个人认为第三题没有做到是最亏的，第六题还蛮惊喜的，因为以为拿左手乘右手是瞎搞

这次为了准备月考做了很多额外题，比如红蓝玫瑰、basketball、变换迷宫，苹果等（每日选做跟不上只能去挑题目），然后也去看和做了很多dp的题目，是一个朋友分享的一系列dp题目，感觉自己快要掌握到dp的诀窍，每次离推出递推只差一点，可恶啊，希望考试dp别那么难，然后希望出bfs，比dfs和dp简单多了

这次考得那么差，三次月考最多AC4,只能祈望机考AC4-5

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>




