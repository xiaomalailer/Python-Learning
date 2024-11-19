# Assignment #8: 田忌赛马来了

Updated 1021 GMT+8 Nov 12, 2024

2024 fall, Complied by <mark>马凱权 元培</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 12558: 岛屿周⻓

matices, http://cs101.openjudge.cn/practice/12558/ 

思路：

先创造好地图（依照题目给的四面环海），岛屿能算周长的邻近部分需为0，即为海，由此检查每个地的上下左右地区


代码：

```python
n,m=map(int,input().split())
map_1=[[0]*(m+2) for _ in range(n+2)]
for i in range(1,n+1):
    tmp=list(map(int,input().split()))
    map_1[i]=[0]+tmp+[0]
cnt=0
for i in range(1,n+1):
    for j in range(1,m+1):
        if map_1[i][j]==1:
            if map_1[i][j-1]==0:
                cnt+=1
            if map_1[i][j+1]==0:
                cnt+=1
            if map_1[i-1][j]==0:
                cnt+=1
            if map_1[i+1][j]==0:
                cnt+=1
print(cnt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20154859.png?raw=true)




### LeetCode54.螺旋矩阵

matrice, https://leetcode.cn/problems/spiral-matrix/

与OJ这个题目一样的 18106: 螺旋矩阵，http://cs101.openjudge.cn/practice/18106

思路：

本来是要用如上一题搬的碰壁方法，但是因为matrix是一开始给定的有点困难，然后参考了答案用了上下左右方法，顺序是先左->右，再上->下，再右->左，最后下->上，然后重复直到所有数字都被记录

后看到一解答觉得甚是奇妙：先用pop去掉和记录第一行，然后旋转矩阵，再重复


代码：

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        
            
        bot = len(matrix)-1
        rig=len(matrix[0])-1
        lef=0
        top=0
        n1=[]

        while True:
            for i in range(lef,rig+1):
                n1.append(matrix[top][i])
            top+=1
            if top>bot:break
            for i in range(top,bot+1):
                n1.append(matrix[i][rig])
            rig-=1
            if lef>rig:break
            for i in range(rig,lef-1,-1):
                n1.append(matrix[bot][i])
            bot-=1
            if top>bot:break
            for i in range(bot,top-1,-1):
                n1.append(matrix[i][lef])
            lef+=1
            if lef>rig:break
            
        return n1
```

另一解法
```python
matrix=[[1,2,3],[4,5,6],[7,8,9]]
result=[]
while matrix:
    result+=matrix.pop(0)
    if matrix:
        matrix=list(zip(*matrix))[::-1]
print(result)
```

对于oj上的题：
```python
n = int(input())  
matrix = [[0] * n for i in range(n)] 
top, bot, lef, rig = 0, n - 1, 0, n - 1  
index = 1  

while top <= bot and lef <= rig:
    
    for i in range(lef, rig + 1):
        matrix[top][i] = index
        index += 1
    top += 1

    
    for i in range(top, bot + 1):
        matrix[i][rig] = index
        index += 1
    rig -= 1

    
    for i in range(rig, lef - 1, -1):
        matrix[bot][i] = index
        index += 1
    bot -= 1

    
    for i in range(bot, top - 1, -1):
        matrix[i][lef] = index
        index += 1
    lef += 1

for i in range(n):
    print(" ".join(map(str, matrix[i])))
```

代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20163831.png?raw=true)



### 04133:垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/

思路：

无奈看了答案，因为已经和题杠了好久好久，有将个人理解注释在代码，主要是最后的处理是，只能允许有一个值是最大的，但是可以有多个相同的最大值，需要用当c和最大值相同时res+1记录，但是一旦发现更大值res重置

代码：

```python
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



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20175854.png?raw=true)




### LeetCode376.摆动序列

greedy, dp, https://leetcode.cn/problems/wiggle-subsequence/

与OJ这个题目一样的，26976:摆动序列, http://cs101.openjudge.cn/routine/26976/

思路：

因为一些原因一直有bug，所以看了答案，整体思路就是一个摆动序列必然是大小大小../小大小大两种可能，据此就找出此二序列再进行比较

代码：

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        down=1
        up=1
        for i in range(1,len(nums)):
            if nums[i]>nums[i-1]:
                up=down+1
            elif nums[i]<nums[i-1]:
                down=up+1
        return(max(down,up))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20192415.png?raw=true)




### CF455A: Boredom

dp, 1500, https://codeforces.com/contest/455/problem/A

思路：

是dp题，状态转移方程是dp[i]=max(dp[i-1],dp[i-2]+arr_2[i]*i)，即要么选该数字，包含该数字-2的（题目规定不能前一），要么不选就维持分数），要注意的是一开始作dp数组时需要使用max(arr_1)+1,如果直接用n，n如果很大会re，dp数组长度最大只需要到数列中最大的号码即可

代码：

```python
n=int(input())
arr_1=list(map(int,input().split()))
arr_2=[0]*(max(arr_1)+1)
for i in arr_1:
    arr_2[i]+=1
dp=[0]*(max(arr_1)+1)
dp[1]=arr_2[1]
 
for i in range(2,max(arr_1)+1):
    dp[i]=max(dp[i-1],dp[i-2]+arr_2[i]*i)
 
print(dp[max(arr_1)])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20203421.png?raw=true)




### 02287: Tian Ji -- The Horse Racing

greedy, dfs http://cs101.openjudge.cn/practice/02287

思路：

田忌赛马思路，正常情况下会先在下等马对上等马时输掉一局，但下一局时田忌的下等马变成了中等马，对上王下等马必赢，最后再由田上等马对王剩下的“上”等马（中等）


代码：

```python
while True:
    n=int(input())
    if n==0:break
    tian_ma=list(map(int,input().split()))
    king_ma=list(map(int,input().split()))
    tian_ma.sort()
    king_ma.sort()
    t1=k1=0
    t2=k2=n-1
    win=0
    lose=0

    while t1<=t2 and k1<=k2:
        if tian_ma[t1]>king_ma[k1]:#若田最慢马快过王最慢马，赢一局
            win+=1
            t1+=1
            k1+=1
        elif tian_ma[t2]>king_ma[k2]:#若田最快马快过王最快马，赢一局
            win+=1
            t2-=1
            k2-=1
        else: #正常情况，则下等马对上等马，输一局，同时王最快马剩中等马，而田最慢马剩中等马
            if tian_ma[t1]<king_ma[k2]:
                lose+=1
            t1+=1
            k2-=1
    print((win-lose)*200)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-12%20214448.png?raw=true)




## 2. 学习总结和收获

终于完成作业！这次作业不会简单，但也不太难，主要是会在细节处理上碰壁。第一题挺简单，第二题就一直写不出，直到参考了上下左右的方法，垃圾炸弹对于要如何计算投点数也是纠结了很久，不过这题让我想起了以前看过的放炸弹在哪里能消灭多少敌人的题，

摆动序列一开始写了比较繁琐去相减又比较之类的然后找不到哪里错，遂参考了答案，认为up down/down up的思路很妙！boredom说实话dp想到还挺有难度的，因为会迷糊的以为该数字的dp就是指选了它的最大分数，这就会很难构造dp，反而应该想成选和不选最大值才是dp

田忌赛马题目很长，感觉读的必要性不大，下等马对上等马，然后双方的马等级抽象的升级/降级...

期中结束了但是还是很忙，因此格外认真去做作业来作为学习计概的主要途径（会努力挤出时间做选做的！），感觉收获蛮大，而且这次会去看各种各样解答，觉得妙的还会再修改自己的or另外记录，因为自己会因为嫌某方法有点难就换成其他啰嗦但好理解的方法，但是让我在无论这次月考还是作业都感到棘手的是写长长的不好debug，也不一定对，所以决定之后都在做完后去看下解答，顺便学习算法

目前认为dp还是很有难度，感觉dp的概念比较抽象，就算写出来在纸上也有点难构造和理解，会试着努力去做相关选做吧！

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>