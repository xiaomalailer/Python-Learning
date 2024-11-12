
# Assignment #6: Recursion and DP

Updated 2201 GMT+8 Oct 29, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy119: 汉诺塔

recursion, https://sunnywhy.com/sfbj/4/3/119  

思路：

如课件里所写，汉诺塔主要有三步（三柱情况下）：

1.将开始柱init中n-1的圆盘移到中间柱temp（借目标柱des达成）

2.将开始柱中第n圆盘移到目标柱（最后柱）

3.将中间柱的圆盘移到目标柱（借开始柱达成）

事实上第一步不可能一步达成，还要推进到边缘条件即1个圆盘的移动，可画图帮助了解

以n=3为例：

第1步：调用 move(2, 'A', 'C', 'B')：移动 2 个盘子，从 A 移到 B，借助 C 作为辅助柱

第2步：调用 move(1, 'A', 'B', 'C')：移动 1 个盘子，从 A 移到 C，借助 B 作为辅助柱

第3步：调用 move(0, 'A', 'C', 'B')：n=0，直接返回。输出 A->C 将盘子 1 从 A 移到 C【1】

第4步：调用 move(0, 'B', 'A', 'C')：n=0，直接返回。输出 A->B,将盘子 2 从 A 移到 B【2】

第5步：调用 move(1, 'C', 'A', 'B')：移动 1 个盘子，从 C 移到 B，借助 A 作为辅助柱

第6步：调用 move(0, 'C', 'B', 'A')：n=0，直接返回。输出 C->B,将盘子 1 从 C 移到 B【3】

第7步：调用 move(0, 'A', 'C', 'B')：n=0，直接返回。输出 A->C,将盘子 3 从 A 移到 C【4】

第8步：调用 move(2, 'B', 'A', 'C')：移动 2 个盘子，从 B 移到 C，借助 A 作为辅助柱

第9步：调用 move(1, 'B', 'C', 'A')：移动 1 个盘子，从 B 移到 A，借助 C 作为辅助柱

第10步：调用 move(0, 'B', 'A', 'C')：n=0，直接返回。输出 B->A,将盘子 1 从 B 移到 A【5】

第11步：调用 move(0, 'C', 'B', 'A')：n=0，直接返回。输出 B->C,将盘子 2 从 B 移到 C【6】

第12步：调用 move(1, 'A', 'B', 'C')：移动 1 个盘子，从 A 移到 C，借助 B 作为辅助柱

第13步：调用 move(0, 'A', 'C', 'B')：n=0，直接返回,输出 A->C,将盘子 1 从 A 移到 C 【7】

第14步：调用 move(0, 'B', 'A', 'C')：n=0，直接返回


代码：

```python
def move(n,init,temp,des):
    if n==0:
        return
    move(n-1,init,des,temp) #借助des的帮忙从init移到temp
    print(f"{init}->{des}") #把init中的第n个圆盘移到des
    move(n-1,temp,init,des) #借助init从temp移到des


n=int(input())

print(2**n-1)
move(int(n),'A','B','C') #A为开始，B为中间柱，c为目标柱
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-29%20223432.png?raw=true)




### sy132: 全排列I

recursion, https://sunnywhy.com/sfbj/4/3/132

思路：

每次在当前位置选择一个未使用的元素，放入当前排列中，然后继续递归地处理剩下的位置，直到排列的长度等于 n，此时得到一个完整排列，记录下来并返回。

返回后进行回溯：递归返回后，移除当前数字，取消标记，以便在下一个循环中尝试其他数字。

代码：

```python

def work_p(idx,n,used,temp,result):
    if idx==n+1:
        result.append(temp[:])
        return
    for i in range(1,n+1):
        if not used[i]:
            temp.append(i)
            used[i]=True
            work_p(idx+1,n,used,temp,result)
            used[i] = False
            temp.pop()


def permutations(n):
    result=[]
    used=[False]*(n+1)
    work_p(1,n,used,[],result)
    for ans in result:
        print(' '.join(map(str,ans)))

n=int(input())
permutations(n)
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-29%20231358.png?raw=true)




### 02945: 拦截导弹 

dp, http://cs101.openjudge.cn/2024fallroutine/02945

思路：

使用dp进行解答，计算先后顺序中每个导弹是第几个被拦截的，

举测试数据：300 207 155 300 299 170 158 65

因为207<300，所以dp[1]=dp[0]+1=2

155<300，dp[2]=dp[0]+1=2,155再与207比，dp[2]=max(dp[2],dp[1]+1]=3

而到了第四个导弹300，300<=300，所以dp[4]=dp[1]+1,但是对于前面的导弹高度都低于它，所以就不理

状态转移方程就是dp[i]=max(dp[i],dp[j]+1),就每个导弹都和前面的导弹比高度，如果允许拦截，dp更新取决于前面可拦截导弹的dp的最大值




代码：

```python
n=int(input())
*a,=map(int,input().split())
dp=[1]*n

for i in range(1,n):
    for j in range(i):
        if a[i]<=a[j]:
            dp[i]=max(dp[i],dp[j]+1)

print(max(dp))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-30%20072840.png?raw=true)




### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：

自从知道dp是什么玩意后，一切都好像简单好多，就是找到状态方程（起码有个目标），虽然不是很好找，去看了课件，那个分格子的方法大大启发了我，于是很快的就可以找到状态方程：

dp[i][j]=max(dp[i-1][j],list_p[i]+dp[i-1][j-list_w[i]])

要理解这个还是强烈建议要看那张图，或者自己根据物品和重量画出格子，用矩阵方式就很清楚了

以测试数据来说，表格应该是这样的：

 磅：0  1  2  3  4

    [0, 0, 0, 0, 0], 

    [0, 0, 0, 0, 3000],  #第一个3000的音响

    [0, 0, 0, 2000, 3000],  #第二个2000的笔记本

    [0, 1500, 1500, 2000, 3500] #第三个1500的吉他

该格的dp等于max（上一个格子的值 ，本格（行）物价+上一个格子的值（加不加取决于磅数够不够））

代码：

```python
n,b=map(int,input().split())
list_p=[0]+list(map(int,input().split()))
list_w=[0]+list(map(int,input().split()))
dp=[[0]*(b+1) for _ in range(n+1)]

for i in range(1,n+1):
    for j in range(1,b+1):
        if list_w[i]>j:
            dp[i][j]=dp[i-1][j]
        elif list_w[i]<=j:
            dp[i][j]=max(dp[i-1][j],list_p[i]+dp[i-1][j-list_w[i]])

print(dp[n][b])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-30%20183316.png?raw=true)




### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754

思路：

得承认自己看了答案，因为实在不知道该怎么写递归，但是思路是有的，就是不能同列，不能斜角，也就是行差和列差不能相等，

然后自己有加了一个print(a+str(col)) 观察递归情况觉得还蛮有意思的，（我认为如果看了答案不了解非常建议使用print去查看递归过程）

比如第一个解，一开始会到‘13524’ 接着却是 '13526' ，因为原本13524后边发现到不能填任何数字就会返回去

然后看了答案后能自己再写一遍并自觉添加注释，还不错

代码：

```python
ans=[]

def queens(a):
    for col in range(1,9): #col列 有8个列
        for i in range(len(a)): #为了比对前面位置的皇后
            if (str(col)==a[i] or abs(len(a)-i)==abs(col-int(a[i]))):#皇后对冲情况：同列 或者 斜对角（行差等于列差）
                break
        else:
            if len(a)==7:
                    ans.append(a+str(col))
            else:
                queens(a+str(col)) #看下一位

queens('')

n=int(input())#输入几组测试数据
for _ in range(n):
    b=int(input())
    print(ans[b - 1]) #从答案组中找到第n个答案输出（-1是因为数组从0开始）

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-31%20092821.png?raw=true)




### 189A. Cut Ribbon 

brute force, dp 1300 https://codeforces.com/problemset/problem/189/A

思路：

这题与小偷背包那题几乎一样思路去做就可以了，我先是用了矩阵的方式去做,但是发现有些麻烦和冗余，遂进行简化，发现只要从当下给定的可切割长度开始计算更佳，因为当下给定的必为1，只要长度-（切割长度）仍然大于或者等于可切割长度就+1



代码：

```python
n,a,b,c=map(int,input().split())
dp=[-float('inf')]*(n+1)
dp[0]=0

for length in [a,b,c]:
    for j in range(length,n+1):
        dp[j]=max(dp[j],dp[j-length]+1) ## 更新 dp[j]，计算当前长度 j 能够切割出的最大段数

print(dp[n])


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-31%20213409.png?raw=true)




## 2. 学习总结和收获

期中季，实在难以赶上每日选做，算法也只是每周看课件跟着学，这周学到了递归和dp，真心觉得dp很好用，学到了这个概念后，面对题目起码不会那么迷茫，会想方法找到所谓的状态方程，找状态方程也很考逻辑能力，有种在解密的感觉，非常好玩

汉诺塔真的好难懂，感觉表面上的步骤那些都很清楚，但是去研究到底是怎样递归调用函数的就很头疼，八皇后那题一开始倒是难倒我了，主要是无从下手，不知道怎么把知道的条件写进去做递归，感觉递归还得再练习，比dp难多了

小偷背包是看课件学到的，特有意思，所以最后一题就做得轻松很多（cf在1031下午一直登不进去。。，延迟了缴交）

又要月考了，希望自己能考好点，最主要是自己也不确定到底掌握了dp和递归，月考前应该会先把每日选做有关这两个的练习下，枚举和greedy自认为问题不大

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>
