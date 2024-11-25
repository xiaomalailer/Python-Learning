### 模板
link：

思路：

####代码
```python

```
学习总结：

### CF455A: Boredom
link：https://codeforces.com/contest/455/problem/A

思路：

####代码
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
学习总结：

