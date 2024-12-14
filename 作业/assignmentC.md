# Assignment #C: ��ζ�ӳ� 

Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by <mark>��PȨ Ԫ��</mark>



**˵����**

1�����ÿ����Ŀ����˼·����ѡ����Դ��Python, ����C++���Ѿ���Codeforces/Openjudge��AC������ͼ������Accepted������д��������ҵģ���У��Ƽ�ʹ�� typora https://typoraio.cn ��������word����AC ����û��AC���������ÿ����Ŀ���»���ʱ�䡣

2���ύʱ�����ύpdf�ļ����ٰ�md����doc�ļ��ϴ����Ҳࡰ��ҵ���ۡ���Canvas��Ҫ��ͬѧ����ͷ���ύ�ļ���pdf��"��ҵ����"�����ϴ���md����doc������

3����������ڽ�ֹǰ�ύ��ҵ����д��ԭ��



## 1. ��Ŀ

### 1115. ȡʯ����Ϸ

dfs, https://www.acwing.com/problem/content/description/1117/

˼·��

������ʾ���߼�����a>=bʱ����ȡ���ֵĻ�ʤ����Ȼ��Ϊ������<1����������ˣ����Ծ�Ҫ��a<bʱ��λ��Ȼ��a//b<2ʱ��ʤ��δ�֣�������һ��ץȡ����ô��һ��ץȡ�Ʊػ��ã�a<b������Ҫ��λ��b,a-b)������Ϊ�ڶ���д������⣬�Ͱ�ץȡ������ż��Ϊ���֣�����Ϊ�����������ж�����˭��ץȡ


���룺

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

����д��Ҳ��
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

�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20144710.png?raw=true)




### 25570: ���

Matrices, http://cs101.openjudge.cn/practice/25570

˼·��

�۲�ɷ��֣�ÿ����ȡ��ʵ��������һ���ã�����һ���� ���������ã��Ͱ�����������һ�����м��㼴��


���룺

```python
n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]

max_sum = 0
layer = 0

while layer < (n + 1) // 2:
    layer_sum = 0

    for i in range(layer, n - layer):# �ϱ߽�
        layer_sum += matrix[layer][i]

    if layer != n - layer - 1:#�±߽�
        for i in range(layer, n - layer):
            layer_sum += matrix[n - layer - 1][i]

    # ���ұ߽�
    for i in range(layer + 1, n - layer - 1):
        layer_sum += matrix[i][layer]  # ���
        layer_sum += matrix[i][n - layer - 1]  # �ұ�

    max_sum = max(max_sum, layer_sum)

    layer += 1

print(max_sum)

```



�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20201614.png?raw=true)




### 1526C1. Potions(Easy Version)

greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

˼·��

�군����ȫ����ʶheapq����gpt��0ѧ�𣬲��˽�ѵ����ʣ��ᶯ̬�Ĵ���Ѷ���֤����С�����Դ�����һ�����ֺ��˸�ҩ����<0���ǾͲ�Ҫ��or�������֮ǰ�ȹ�����Сֵ���Ǿ����滻���������֮����¸��ࣩ

���룺

```python
import heapq

def max_p(n,potion):
    health_now=0
    cnt=0
    heap=[]

    for p in potion:
        if health_now+p>=0:
            heapq.heappush(heap,p)#���µ�ǰҩˮ
            health_now+=p
            cnt+=1
        elif heap and p>heap[0]:#���º�����ֵΪ���Ҵ�����ǰ���µ���Сֵ
            health_now+=p-heapq.heappop(heap)
            heapq.heappush(heap,p)
    return cnt

n=int(input())
potion=list(map(int,input().split()))
print(max_p(n,potion))

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20220930.png?raw=true)




### 22067: ���ٶ���

����ջ��http://cs101.openjudge.cn/practice/22067/

˼·��

��ʾ������Ҫ�ø���ջ�����˲�ע�⣬���̱�ջ������ջ������ά����Сֵ�õģ�֮��ע��û����popʱҲ�Ѹ���ջpopҲ����ɴ���ά��������ʵ���ǱȽ�����push�ĺ�֮ǰջ������Сֵ����˭��С˭��ջ��ջ����


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20230230.png?raw=true)




### 20106: ��ɽ·

Dijkstra, http://cs101.openjudge.cn/practice/20106/

˼·��

����а��bfs�������������ȥ�˽�dijkstra��ԭ����potion�����heapq������Ƚ�����㣬��Ҫ��ȥ������С�ģ�����pqȥά���Ѷ�����Сֵ��һ�ֲ�ͬ�汾��bfs�ĸо�

���룺

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
                heappush(pq, (cost + new_cost, nx, ny))#ȷ��cost��С�ڶѶ�

    return 'NO'


# ���봦��
m, n, p = map(int, input().split())
maze = [input().split() for _ in range(m)]

for _ in range(p):
    x1, y1, x2, y2 = map(int, input().split())
    if maze[x1][y1] == '#' or maze[x2][y2] == '#':
        print('NO')
    else:
        print(dijkstra(x1, y1, x2, y2))

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20230230.png?raw=true)




### 04129: �任���Թ�

bfs, http://cs101.openjudge.cn/practice/04129/

˼·��

�������¿�ǰ�����ˣ����Ի�д����Ȼ��ʵ����һ��ʼ������ʱ���ǿ��˺ܾÿ����ģ��ؼ�������temp=(time+1)%k�ϣ������ж�ʯͷ�᲻����ʧ��������Ѿ����ʹ� (temp, nx, ny) ���״̬����ζ��������ͬ��ʱ�����ں���ͬ��λ�����Ѿ������˴������沽�趼��һ���ģ�Ҳ����˵����ĳ�㴦ʱʱ����t����t+k��һ�����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-12-10%20222501.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

�군���군�ˣ������������ҵ���Ѷ�ҲҪ�군�����Ⱦ���potions����ɽ·�����õ��ѣ�������ѹ��ûȥѧ����������˺þõ���������ά���Ѷ��ģ��о���������ò�

�о��Լ���һ����Ŀ�ͻ�ܻŲ�֪����ô�죬�����������ʾ�ͻ���˼·ȥ����0��1��1��100�Ѻöࡣ����

��������ʱ�䣬ץ����ϰdp������ʱ����ʻᱩ�������ˣ�ϣ����Ŀ��Ҫ��ô�ϸ�

cheatsheet���Դ�ӡ����Ӧ�ûὫһЩ����������¼����������ʱ��ȥ����У�����ϣ���Լ���AC4���ϣ�����~~

<mark>�����ҵ��Ŀ�򵥣��з������ϰ��Ŀ�����磺OJ���Ƹ�2024fallÿ��ѡ������CF��LeetCode����ȵ���վ��Ŀ��</mark>