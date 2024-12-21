# Assignment #D: ʮȫʮ�� 

Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by <mark>��PȨ Ԫ��</mark>



**˵����**

1�����ÿ����Ŀ����˼·����ѡ����Դ��Python, ����C++���Ѿ���Codeforces/Openjudge��AC������ͼ������Accepted������д��������ҵģ���У��Ƽ�ʹ�� typora https://typoraio.cn ��������word����AC ����û��AC���������ÿ����Ŀ���»���ʱ�䡣

2���ύʱ�����ύpdf�ļ����ٰ�md����doc�ļ��ϴ����Ҳࡰ��ҵ���ۡ���Canvas��Ҫ��ͬѧ����ͷ���ύ�ļ���pdf��"��ҵ����"�����ϴ���md����doc������

3����������ڽ�ֹǰ�ύ��ҵ����д��ԭ��



## 1. ��Ŀ

### 02692: �ٱ�����

brute force, http://cs101.openjudge.cn/practice/02692

˼·��

ʵ�ڲ�֪������ô���֣����˽��ֻ��˵�Լ���ô��ô��û�뵽������Ǽٱң���Ϊ�غ�������������ؾ�����������up(���ϣ��������ұ�down�����û�зž�even���������β������������������Ǽٱң���ļٱ����ƣ�һ�������ÿ���Ϊ�ٱҵĿ����Լ���

���룺

```python


def check_coin(coins):
    for coin in 'ABCDEFGHIJKL':
        if all( (coin in s[0] and s[2]=='down' ) or \
            (coin in s[1] and s[2]=='up' ) or \
                (coin not in s[0]+s[1] and s[2]=='even') for s in coins):
            print('{} is the counterfeit coin and it is {}.'.format(coin, 'light'))
            break
        if all((coin in s[0] and s[2]=='up' ) or \
            (coin in s[1] and s[2]=='down' ) or \
                (coin not in s[0]+s[1] and s[2]=='even') for s in coins):
            print('{} is the counterfeit coin and it is {}.'.format(coin, 'heavy'))
            break

n=int(input())
for _ in range(n):
    coins=[[],[],[]]
    for i in range(3):
        coins[i]=input().split()
    check_coin(coins)


```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20172356.png?raw=true)




### 01088: ��ѩ

dp, dfs similar, http://cs101.openjudge.cn/practice/01088

˼·��

����Ҳ�ǽ���������һ�⣬ԭ����д��һȦ����Ȧ��100001)ȥ������ͼ���Է��ܳ�ȥ����������ʱ����ú��ޱ�Ҫ����������dfs��ֻ��˵��Ϊ��TLE��������lru_cache

���룺

```python
from functools import lru_cache

move=[(0,1),(0,-1),(1,0),(-1,0)]

@lru_cache(None)
def dfs(x,y):
    ans=1
    for dx,dy in move:
        nx,ny=x+dx,y+dy
        if 0<=nx<R and 0<=ny<C and maze[nx][ny]<maze[x][y]:
            ans=max(ans,dfs(nx,ny)+1)
    return ans

R,C=map(int,input().split())
maze=[list(map(int,input().split())) for _ in range(R)]
res=0
for i in range(R):
    for j in range(C):
        res=max(res,dfs(i,j))
print(res)
```



�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20162658.png?raw=true)




### 25572: �з��Ģ��

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

˼·��

һλ�ſ�ѧ�������ҵģ�����м�����Ҫ�ƶ���ÿ���㶼ȥ�ж����У��������У�����ͱ����һ��ܼ򵥵�bfs��֮ǰ����һ�Σ����ǲ����Ž��д��type1 type2���ֺ�����ģ������û��Ҫ

���룺

```python
from collections import deque

move = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def bfs(s_x1, s_y1, s_x2, s_y2):

    if not ((abs(s_x1 - s_x2) == 1 and s_y1 == s_y2) or (s_x1 == s_x2 and abs(s_y1 - s_y2) == 1)):
        return False

    q = deque()
    q.append((s_x1, s_y1, s_x2, s_y2))
    inq = set()
    inq.add((s_x1, s_y1, s_x2, s_y2))

    while q:
        x1, y1, x2, y2 = q.popleft()

        if maze[x1][y1] == 9 or maze[x2][y2] == 9:
            return True

        for dx, dy in move:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy

            if 0 <= nx1 < n and 0 <= ny1 < n and 0 <= nx2 < n and 0 <= ny2 < n:
                if maze[nx1][ny1] != 1 and maze[nx2][ny2] != 1:
                    if (nx1, ny1, nx2, ny2) not in inq:
                        inq.add((nx1, ny1, nx2, ny2))
                        q.append((nx1, ny1, nx2, ny2))

    return False



n = int(input())
maze = [list(map(int, input().split())) for _ in range(n)]
a = []
for i in range(n):
    for j in range(n):
        if maze[i][j] == 5:
            a.append([i, j])


if len(a) == 2:
    result = bfs(a[0][0], a[0][1], a[1][0], a[1][1])
    print('yes' if result else 'no')
else:
    print('no')

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20161344.png?raw=true)




### 27373: �������

dp, http://cs101.openjudge.cn/practice/27373/

˼·��

��Ҫ��str��intת������debug��Щʱ�䣬���⻹������������ѣ�����С͵����������д����ǰ���е��ѵ���Ҫ�������У����˽����뵽������λ����ȥ����


���룺

```python

def c(s):
    if s=='':
        return 0
    else:
        return int(s)

m=int(input()) #λ��
n=int(input()) #��������
arr=input().split()

#����λ���ִ�С��������
for i in range(n):
    for j in range(n-i-1):
        if arr[j]+arr[j+1]>arr[j+1]+arr[j]:
            temp=arr[j]
            arr[j]=arr[j+1]
            arr[j+1]=temp
weight=[]
for i in arr:
    weight.append(len(i))

dp=[['']*(m+1) for _ in range(n+1)]# dp[i][j] i forλ�� j for ����

for s in range(1,n+1): #��û��ȡ�κ����֣���Ϊ�����
    dp[s][0]=''
for s in range(m+1): #��λ��Ϊ0Ҳ����
    dp[0][s]=''
#��С͵����
for i in range(1,n+1):
    for j in range(1,m+1):
        if weight[i-1]>j:#������λ��������λ��������ȡ
            dp[i][j]=dp[i-1][j]
        else:
            dp[i][j]=str(max(c(dp[i-1][j]),int(arr[i-1]+dp[i-1][j-weight[i-1]])))
print(dp[n][m])

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20182014.png?raw=true)




### 02811: Ϩ������

brute force, http://cs101.openjudge.cn/practice/02811

˼·��

̫���ˣ����Գ�������������ɣ�����Ŀ��ʵһ��ʼҲû��ô���ö���ֱ�����˽���Ŷ��ԭ����Ҫ����һ�в�ͬ״̬�����һ�ɵ�����ľ͸��ݵ�һ����״̬ȥ�������У�Ȼ���ֵ�һ��ȫ��0

gpt����������product�����У�

����
```
��ť��ţ�1  2  3  4  5  6
����״̬��0  0  0  0  0  0  (����)
         0  0  0  0  0  1  (����6��)
         0  0  0  0  1  0  (����5��)
          ...
         1  1  1  1  1  1  (ȫ��)
```

���룺

```python
from copy import deepcopy
from itertools import product


TOGGLE = {0: 1, 1: 0}# ȫ�ֱ�������ת״̬ӳ��


def toggle_lights(matrix, i, j):
    matrix[i][j] = TOGGLE[matrix[i][j]]  # ��ǰ��
    matrix[i - 1][j] = TOGGLE[matrix[i - 1][j]]  # �Ϸ���
    matrix[i + 1][j] = TOGGLE[matrix[i + 1][j]]  # �·���
    matrix[i][j - 1] = TOGGLE[matrix[i][j - 1]]  # �󷽵�
    matrix[i][j + 1] = TOGGLE[matrix[i][j + 1]]  # �ҷ���


def solve_lights():

    matrix_backup = [[0] * 8] + [[0] + list(map(int, input().split())) + [0] for _ in range(5)] + [[0] * 8]

    # �������е�1�д���������2^6�֣�
    for test_case in product(range(2), repeat=6):
        matrix = deepcopy(matrix_backup)  # �����ʼ����
        triggers = [list(test_case)]  # ��¼������������1��Ϊ��ǰ�����ķ���

        # ������2����5�У�������һ��״̬�������°�ť
        for i in range(1, 6):
            for j in range(1, 7):
                if triggers[i - 1][j - 1]:
                    toggle_lights(matrix, i, j)

            triggers.append(matrix[i][1:7])# ��¼��ǰ�еĴ����������һ�еİ�ť����״̬��

        # �жϵ�6���Ƿ�ȫ��Ϩ��
        if matrix[5][1:7] == [0] * 6:
            # ����������������������һ�У�
            for trigger in triggers[:-1]:
                print(" ".join(map(str, trigger)))
            return


solve_lights()

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20211643.png?raw=true)




### 08210: ����������

binary search, greedy, http://cs101.openjudge.cn/practice/08210/

˼·��

�Ƚ������Ծ������Ϊ·�̵�һ�루��������˵���������㿴Ҫ�Ƴ�����ʯͷ���ܴ����̾��룬����Ƴ�����ʯͷ����������������Ծ������������У�������һ�룻��������Ƴ�ʯͷ�������ַ���������Ծ����

���������˳������Ƶ� 04135:�¶ȿ��� �Լ����ˣ��������������¶ȿ���������Сֵ��ѡȡleft���Ѷ�

���룺

```python
def binary_stone(s):
    M=0
    s_now=0
    for i in range(1,n+2):
        if stone[i]-s_now<s:
            M+=1
        else:
            s_now=stone[i]

    if M>m:
        return True
    else:
        return False



L,n,m=map(int,input().split())
stone=[0]
for _ in range(n):
    stone.append(int(input()))
stone.append(L)

left=0
right=L+1
ans=0
while left<right:
    mid=(left+right)//2
    if binary_stone(mid):
        right=mid
    else:
        left=mid+1
        ans=mid
print(ans)

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-17%20215806.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

�����������ҵ�������ϻ�Ҫ�군��״̬����������Ŀ�������ף���ε��Ǳ�����bruteforce��greedy���úòң���Ҫ��������������Ŀ�������ܹ��죬��ʱ���˺ܾö���������ĿҪ�󣬵���һ�������ԭ���ֺܼ򵥣�������Ŀ��ǧ��ٹֲ�֪��ô������

���ܴ�Ž�cheatsheet������ˣ�����һЩ���õ��﷨��������Ҳ��һЩ��Ŀ�Ž�ȥ���ο���ϣ������ʱ������������Ҳϣ���ϻ���Ĳ�Ҫ��ô�ѣ���Ϊ�ҿ��˱��Է��ָ�Ҫ��~

Ŀǰ�Լ����������169�⣬����ÿ��ѡ��+��ҵ��������ֻ��ѡ���Ե�ȥ��ÿ��ѡ����

<mark>�����ҵ��Ŀ�򵥣��з������ϰ��Ŀ�����磺OJ���Ƹ�2024fallÿ��ѡ������CF��LeetCode����ȵ���վ��Ŀ��</mark>