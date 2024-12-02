# Assignment #10: dp & bfs

Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by <mark>��PȨ Ԫ��</mark>



**˵����**

1�����ÿ����Ŀ����˼·����ѡ����Դ��Python, ����C++���Ѿ���Codeforces/Openjudge��AC������ͼ������Accepted������д��������ҵģ���У��Ƽ�ʹ�� typora https://typoraio.cn ��������word����AC ����û��AC���������ÿ����Ŀ���»���ʱ�䡣

2���ύʱ�����ύpdf�ļ����ٰ�md����doc�ļ��ϴ����Ҳࡰ��ҵ���ۡ���Canvas��Ҫ��ͬѧ����ͷ���ύ�ļ���pdf��"��ҵ����"�����ϴ���md����doc������

3����������ڽ�ֹǰ�ύ��ҵ����д��ԭ��



## 1. ��Ŀ

### LuoguP1255 ��¥��

dp, bfs, https://www.luogu.com.cn/problem/P1255

˼·��

����������bfs˼����һ��ʼ����ģ�壬����ֻ��Ҫ��pos����pos�����ݣ�����ͺã�Ȼ��֪������ô��¼·������������gpt����ways���д��棬��������������ߵ� pos+1���� pos+1 <= n������͸��� ways[pos+1]�����Ӵ� pos �� pos+1 ��·����������pos+2һ����ways[n] �ͻ��Ǵӵ�0�׵���� n �׵����в�ͬ·����

������ʵ��dp���죬״̬�����Ǹý���·�����͵���n-1��n-2����·�����ͣ���ע�����dp[0]��dp[1]=1

���룺

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

�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-26%20195146.png?raw=true)




### 27528: ��̨��

dp, http://cs101.openjudge.cn/practice/27528/

˼·��

�����һ�������2�����ھ͸㶨�ڶ��⣬��Ϊ����һ�����Ķ��ĵط��ǲ�ֹ��n-1��n-2�ִ�����Դ�n-j�ִ�����Ǵ�n-n=0��һ����λ

���룺

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



�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-26%20202130.png?raw=true)




### 474D. Flowers

dp, https://codeforces.com/problemset/problem/474/D

˼·��

����һСʱ���벻����������ֿ���1Сʱ�����⣬һ��ʼҲû����MOD������ʲô��dpһ��ʼ��֪����ô�裬���˽������˺�һ���ӣ���ʵ���ǰѵ�ǰ�Ǻ컨���ǻƻ�����������ܣ�����Ǻ컨��ǰ��ʲô�������ԣ�����ǻƻ�����Ҫk-1���ƻ�����


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-27%20214432.png?raw=true)




### LeetCode5.������Ӵ�

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

˼·��

����һ��ʼ��û�뵽Ҫ��ô��dp�������������˶��ֲ������ڻ������зǳ����ã��������������Ҫ��dp���ܾ���i=left��j=right����ȥ��д���ɣ�������Ĳ��������׺��Եľ����м�Ҳ�ظ���abba�����Ľṹ�ģ����ҷ�ʽ�б���aba

���룺

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
        l2,r2=find_palindrome(i,i+1) #Ӧ�Ի��������м�ǵ��������abba
        if r1-l1>end-start:
            start,end=l1,r1
        if r2-l2>end-start:
            start,end=l2,r2
    return s[start:end+1]

s=input()
print(longest_palindrome(s))
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-27%20221357.png?raw=true)






### 12029: ˮ���߾�

bfs, dfs, http://cs101.openjudge.cn/practice/12029/

˼·��

����Ҫ��dfs�⣬�Ѿ�д���ˣ���Ŀ�������ݺ��Լ����Ҳ���ˣ�����oj�Ͼ���һֱre�����˴𰸾͸�һ�㻹��re�����ĵ�����99%�ӽ��˶������鷳������Ϊʲô������ô�鷳��������һ��ܼ򵥵���dfs��bfs�⣬�ҿ���bfs��Ҳ��һ���鷳

���룺

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

bfs�����Լ����ԣ�
```python
from collections import deque
import sys
input = sys.stdin.read

# �ж������Ƿ���Ч
def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n

# �����������ģ��ˮ��
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

# ������
def main():
    data = input().split()  # ���ٶ�ȡ������������
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

�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-28%20000403.png?raw=true)




### 02802: С��Ϸ

bfs, http://cs101.openjudge.cn/practice/02802/

˼·��

��˵˼· �ͼ򵥵����������ߵ�Ŀ�ĵأ�ȡ���·�����鷳���ǲ�֪��ô����������������ܹ����Ǿ���RE/WA��ֻ�����Ųο����޸ģ�����enumerate���Ҿ��ùؼ�������inq������visited����Ҫ������Ƿ���i���ܴ���ܾ���������µ��Ҳ�ͨ��


���룺

```python
from collections import deque

move = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def bfs(start, end, m, n, maze):
    global ans
    inq = set()

    q = deque()
    n_x, n_y = start
    q.append((n_x, n_y, -1, 0))  # �����ʼΪ-1������Ϊ0


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
                elif maze[nx][ny] != 'X':  # ��Ϊ�ϰ���
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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-11-28%20165547.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

�ܽᣬ�����ҵ���Ѱ���bfs��dfs��Ŀ����������ѣ��׸�ģ����Կ�д�������鷳����̫����Ҫdebug��ĵÿ��ð��죬�����Ҳ�ö��պþá�����

�ѵĵط���dp���Լ����ǲ���д�����������ҵ�󲿷ֶ���Ѱ��ο��𰸻���ai��æ��ϣ�����Ա���ô��

���Լ�ȥ����23������ϻ�����AC4��������һ����bfs���Կ���ģ�壬����Ӧ�û��bfs����cheatsheet��������������һ��dpһ����permutationʵ�ڲ��ᣬϣ�����Բ�����ô�ѿ�������ĿǰҲ��cheatsheetд���﷨�Ĳ��֣����ž���д���㷨�ͳ���Ҫ��һЩ�����



<mark>�����ҵ��Ŀ�򵥣��з������ϰ��Ŀ�����磺OJ���Ƹ�2024fallÿ��ѡ������CF��LeetCode����ȵ���վ��Ŀ��</mark>