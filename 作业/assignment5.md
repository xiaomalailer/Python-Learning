# Assignment #5: Greedy穷举Implementation

Updated 1939 GMT+8 Oct 21, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 04148: 生理周期

brute force, http://cs101.openjudge.cn/practice/04148

思路：

思路就是从给定天数的下一天开始找同时满足n-p是23的倍数（n距离p一个周期，到达最高峰），n-e是28的倍数，和n-i是33的倍数,找到了就返回n-d即为答案，此法算是枚举，有点耗时

耗时约20分钟

代码：

```python
def peak(p, e, i, d):
    phy_cycle = 23
    emo_cycle = 28
    int_cycle = 33
    n = d + 1
    while True:
        if (n - p) % phy_cycle == 0 and (n - e) % emo_cycle == 0 and (n - i) % int_cycle == 0:
            return n - d
        n += 1


cnt=0
while True:
    cnt+=1
    p, e, i, d = map(int, input().split())
    if p == -1 and e == -1 and i == -1 and d == -1:
        break
    result = peak(p, e, i, d)
    print(f"Case {cnt}: the next triple peak occurs in {result} days.")

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20153101.png?raw=true)




### 18211: 军备竞赛

greedy, two pointers, http://cs101.openjudge.cn/practice/18211

思路：

花了很长时间改正，大概就是排序好后，就从最小的买起，如果连最小的也买不起也就是cnt=0，那就break；买了之和剩的钱不够继续买，就卖最大的，再回来继续买，不知道是不是就是二分法，然后比较麻烦的就是如果买卖之间最后是卖是错的，所以加了一个max，来保留每次买的时候cnt的最大值

耗时：33min

代码：

```python
p=int(input())
weapon_p=list(map(int,input().split()))
weapon_p=sorted(weapon_p)
cnt=0
start=0
end=len(weapon_p)-1
out=0
while start<=end:
    if p>=int(weapon_p[start]):
        cnt+=1
        p-=int(weapon_p[start])
        start+=1
        out=max(cnt,out)
    elif cnt==0:
        break
    else:
        cnt-=1
        p+=int(weapon_p[end])
        end-=1

print(out)
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20162148.png?raw=true)




### 21554: 排队做实验

greedy, http://cs101.openjudge.cn/practice/21554

思路：

这题倒是不难，要想到用tuple就可以了，而且从测试数据就可以看出排列方式就是根据实验时长长短，然后等待时间稍微拿笔纸算了下就可以得出计算方法，即第一位的实验时长是后边每个人都要等的，*(n-1),以此类推

耗时 20min（不知道把时间花哪儿了）

代码：

```python
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20210252.png?raw=true)



### 01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/

思路：

这题一看就得用字典了，然后先从haab的日历算出总天数，就是年份*365天，加上月份（1个月20天）+天数，然后转换方法就是那个number是每13重复一次，中间的词是每20重复

耗时 22min


代码：

```python
n=int(input())
Haab={'pop':0,'no':1,'zip':2,'zotz':3, 'tzec':4, 'xul':5, 'yoxkin':6, 'mol':7, 'chen':8, 'yax':9, 'zac':10, 'ceh':11, 'mac':12, 'kankin':13,'muan':14, 'pax':15, 'koyab':16, 'cumhu':17,'uayet':18}
Tzolkin={0:'imix', 1: "ik", 2: "akbal", 3: "kan", 4: "chicchan", 5: "cimi", 6: "manik", 7: "lamat", 8: "muluk", 9: "ok", 10: "chuen", 11: "eb", 12: "ben", 13: "ix", 14: "mem", 15: "cib", 16: "caban", 17: "eznab", 18: "canac", 19: "ahau"}
print(n)

for _ in range(n):
    day,month,year=input().split()
    day=int(day.rstrip('.'))
    total_day=int(year)*365+Haab[month]*20+day
    print(total_day%13+1,Tzolkin[total_day%20],total_day//260)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20232535.png?raw=true)




### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C

思路：

这题幸好是CF，能够透过测试数据慢慢修改我代码的漏洞，主要是第一个和最后一个其实是一定能砍的，然后还有一个要注意的地方是如果砍倒的某棵树是向右倒，那么在判断下棵树能不能向左倒的时候prv就是指的就是上一个树的位置应该包含被砍倒的距离

耗时30+min

代码：

```python
n = int(input())
tree = []


for _ in range(n):
    a, b = map(int, input().split())
    tree.append((a, b))

prv = float('-inf')
cnt = 0

for i in range(n):

    if prv + tree[i][1] < tree[i][0]:
        cnt += 1
        prv = tree[i][0]

    elif i < n - 1 and tree[i][1] + tree[i][0] < tree[i+1][0]:
        cnt += 1
        prv = tree[i][0]+tree[i][1]
    elif i==n-1:
        cnt+=1
    else:
        prv = tree[i][0]

print(cnt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-23%20002606.png?raw=true)




### 01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/

思路：

这一题花了很长时间，且一直RE，先是空行问题，然后还有就是那个y不能大于d的问题一开始忽略到了

基本思路是首先雷达只能放在海岸线上，所有雷达距离岛最远的距离，自然是呈一个直角三角形斜边方向的距离，所以采用d^2-y^2，来确定一个岛屿能被雷达监测到，雷达所需放置的区间，然后判断每个岛屿的区间（右部分）是否在上个检测岛屿的区间（左部分），是则可共用同个雷达，否则不同雷达

代码：

```python
import math
ctt=0
while True:
    ctt+=1
    n,d=map(int,input().split())
    if n==0 and d==0:
        break
    flag=True

    cnt=0
    islands=[list(map(int,input().split())) for _ in range(n)]
    for i in range(n):
        if islands[i][1]>d:
            flag=False
            break
        islands[i]=[islands[i][0]-math.sqrt(d*d-islands[i][1]**2),islands[i][0]+math.sqrt(d*d-islands[i][1]**2)]
    if not flag:
        print(f"Case {ctt}: -1")
        input()
        continue
    else:
        islands.sort(key=lambda x:x[1]) #对右部分排序
        sta=float('-inf')
        for i in range(n):
            if sta<islands[i][0]:
                cnt+=1
                sta=islands[i][1]
    print(f"Case {ctt}: {cnt}")
    input()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-23%20235825.png?raw=true)




## 2. 学习总结和收获

这次作业一天内基本可以搞定前5题，题目都很好，练习了排序、贪心，字典等，也很考个人的数学逻辑能力，第6题主要是要花很长时间不断试错才能慢慢改到AC，因为最近很忙，每日选做没怎么碰，老师的课件也还需要时间消化，幸好有作业能让我在主动去做的同时能学到，否则我可能要拖到很迟才在每日选做做到对应部分

这次的作业有个困难的就是要有扎实的数学基础，才能进一步解题，否则就算会算法也较难回答

待应付完期中必全力追赶，作业难度放大我觉得其实蛮好的，可以使我“被动”地学习，相比较前几次，这次倒是真有学习到

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>