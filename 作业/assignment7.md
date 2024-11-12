
# Assignment #7: Nov Mock Exam立冬

Updated 1646 GMT+8 Nov 7, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）⽉考： AC6 <mark>4</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E07618: 病人排队

sorttings, http://cs101.openjudge.cn/practice/07618/

思路：

这是最近新学的lambda用法，-x[0]表示按年龄逆序排列，x[2]代表的是i其实是index，一样年龄便按这个顺序排，当然60岁以下的不做处理直接输出即可

代码：

```python
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-07%20212904.png?raw=true)




### E23555: 节省存储的矩阵乘法

implementation, matrices, http://cs101.openjudge.cn/practice/23555/

思路：
 
这题的思路是先创造两个n*n*矩阵，其中元素全为0，再根据输入进行更改，然后运用矩阵相乘的方法 行*列 进行运算，如果运算结果不为0便输出


代码：

```python
n,m1,m2=map(int,input().split())
arr_1=[[0]*n for _ in range(n)]
arr_2=[[0]*n for _ in range(n)]
for _ in range(m1):
    a,b,c=map(int,input().split())
    arr_1[a][b]=c
for _ in range(m2):
    a,b,c=map(int,input().split())
    arr_2[a][b]=c

result=[[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        for k in range(n):
            result[i][j] += arr_1[i][k] * arr_2[k][j]
        if result[i][j]!=0:
            print(f"{i} {j} {result[i][j]}")
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-07%20213357.png?raw=true)




### M18182: 打怪兽 

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

思路：

因为代码写得有些长，所以加了注释，方便理解。大体思路为累计当下时刻的伤害前m个加起来再攻击，如果怪兽没死，就进入下个时刻的累计，值得注意的是，触发攻击条件是碰到下个时刻的伤害，所以最后一个时刻的技能要额外处理


代码：

```python
def process(n, m, b, pwe):
    # 按时间排序，时间相同的按伤害逆序排序
    pwe.sort(key=lambda x: (x[0], -x[1]))
    current_time = 0
    current_blood = b
    current_power = []

    for time, damage in pwe:
        # 如果当前技能的时间超过了当前时间点，则在该时刻进行一次结算
        if time > current_time:
            if current_power:
                # 选取当前时刻可以使用的最大 m 个技能
                current_power.sort(reverse=True)
                total_damage = sum(current_power[:m])
                current_blood -= total_damage
                if current_blood <= 0:
                    return current_time  # 怪兽死亡的时间
            # 更新到新的时间点，并清空当前技能列表
            current_time = time
            current_power = []

        # 将当前技能加入当前时刻的可用技能列表
        current_power.append(damage)

    # 处理最后一个时刻技能
    if current_power:
        current_power.sort(reverse=True)
        total_damage = sum(current_power[:m])
        current_blood -= total_damage
        if current_blood <= 0:
            return current_time

    return -1  # 如果怪兽未被击败，返回 -1


n_cases = int(input())
for _ in range(n_cases):
    n, m, b = map(int, input().split())
    pwe = [tuple(map(int, input().split())) for _ in range(n)]

    result = process(n, m, b, pwe)
    if result == -1:
        print('alive')
    else:
        print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-08%20081849.png?raw=true)




### M28780: 零钱兑换3

dp, http://cs101.openjudge.cn/practice/28780/

思路：

是dp的题目，dp状态方程的设立思路是算每个欲凑总金额需要用到的最少硬币数，感觉做多了dp就能想出来该怎么去写

代码：

```python
n,m=map(int,input().split())
arr_1=list(map(int,input().split()))
dp=[float('inf')]*(m+1)
dp[0]=0

for i in arr_1:
    for j in range(i,m+1):
        dp[j]=min(dp[j],dp[j-i]+1)

if dp[m]==float('inf'):
    print('-1')
else:
    print(dp[m])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-07%20213916.png?raw=true)




### T12757: 阿尔法星人翻译官

implementation, http://cs101.openjudge.cn/practice/12757

思路：

这题一直we的原因在于不同于thousand和million，hundred不是一个可以当作终止点的条件，比如one hundred thousand，需要先算得100*1000,而算到1000就可以终止，再算接下来部分

代码：

```python
eng_dict = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000, 'million': 1000000
}

english_v = list(input().split())

sum = 0
temp = 0
negative = False

for i in english_v:
    if i == 'negative':
        negative = True
    elif i in eng_dict:
        value = eng_dict[i]
        if value == 100:
            temp *= value
        elif value == 1000 or value == 1000000:
            temp *= value
            sum += temp
            temp = 0
        else:
            temp += value

sum += temp


if negative:
    sum *= -1

print(sum)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-08%20090755.png?raw=true)




### T16528: 充实的寒假生活

greedy/dp, cs10117 Final Exam, http://cs101.openjudge.cn/practice/16528/

思路：

本来想用dp但是弄不出状态方程，遂改用greedy，思路是按结束时间顺序排，然后如果开始时间晚于上个结束时间就+1


代码：

```python
n=int(input())
act=[]
for _ in range(n):
    srt,end=map(int,input().split())
    act.append((srt,end))

act.sort(key=lambda x: x[1])
max_act=0
lst_end=-1
for start,end in act:
    if start>lst_end:
        max_act+=1
        lst_end=end
print(max_act)

```
对于这题看了dp的解答觉得很好：
```python
n = int(input())
event = [-1] * 61
for x in range(n):
    start, end = map(int, input().split())
    event[end] = max(event[end], start)
    
dp = [0]*61
for t in range(61):
    if event[t] == -1:
        dp[t] = max(dp[t], dp[t-1])
    else:
        dp[t] = max(dp[t], dp[t-1], dp[event[t] -1] +1)	#不取这次活动，取这次活动。类似背包

print(dp[60])
```


代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-07%20215147.png?raw=true)




## 2. 学习总结和收获

先总结此次月考表现，差强人意，这次月考题目说实话都比作业简单，但是本身状态太拖，一道题不知道怎么地可以对着发呆好久，AC4道，说实话不满意哈哈，在最后20分钟内疯狂赶打怪兽那题，结果还是没成功，这次我的顺序是先做了第一二题，然后就跑去做第4题，因为是dp题，然后再做第6题（也是因为看到可用dp），才去做第三题，然后就卡着了，过后做作业时发现其实第五题简单多了

不过得鼓励自己下，这次月考有学以致用，把先前做题做作业学到的都有用到些，只是要进步的地方是思考要快些，然后不要烦些低级错误，致使自己花了很长时间debug

期中季差不多结束了，可以开始冲题了，上次AC3 这次AC4，期待下次AC5？，然后AK!(还是想太多了）

感觉学习到的算法、语法差不多了，可以开始写cheatsheet以防之后忘记些基础算法/语法，这次月考有一些时间就在修改语法


<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>
