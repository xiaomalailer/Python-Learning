
## 1. ��Ŀ

### 02733: �ж�����

http://cs101.openjudge.cn/practice/02733/



˼·��
�ܱ�3200�����ķ����꣬�ܱ�100�������ǲ��ܱ�400��������Ҳ�����꣬�����ܱ�4������Ϊ���꣬����Ϊƽ��




##### ����

```python
year=int(input()) 
if year%3200==0:
	print('N')
elif year%400!=0 and year%100==0:
	print('N')
elif year%4==0:
	print('Y')
else:
	print('N')
```



�������н�ͼ ==�����ٰ�����"Accepted"��==

![q1](https://github.com/xiaomalailer/img/blob/main/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-10%20215207.png?raw=true)




### 02750: ����ͬ��

http://cs101.openjudge.cn/practice/02750/



˼·������ŵ������ܱ�4��������ô���������ȫΪ���ӣ�������ܱ�4�������ܱ�2���������������������һֻ��



##### ����

```python
a=int(input())
if a%4==0:
    print(int(a/4),int(a/2))
elif a%2==0:
    print(int((a+2)/4),int(a/2))
else: print("0 0")
```



�������н�ͼ ==�����ٰ�����"Accepted"��==

![q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-11%20202356.png?raw=true)




### 50A. Domino piling

greedy, math, 800, http://codeforces.com/problemset/problem/50/A



˼·���۲췢�֣���M*N��СС��n��dominoes��С����Ϊn-1



##### ����

```python
M, N = map(int, input().split())
n=0
result=0
while result!=M*N:
    n += 1
    result=(M*N)%(n*2)
 
print(n-1)
```



�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==

![q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-11%20202952.png?raw=true)



### 1A. Theatre Square

math, 1000, https://codeforces.com/problemset/problem/1/A



˼·���˽���Ŀ�� ������������������һ��flagstones�߳����ֱ�����ȡ��



##### ����

```python
import math
 
def min_flagstones(n, m, a):
    
    rows = math.ceil(n / a) #����ȡ��
    cols = math.ceil(m / a)
    
    return rows * cols
 
n,m,a=map(int,input().split())
print(min_flagstones(n, m, a))
```



�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==

![q4](https://github.com/xiaomalailer/img/blob/main/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-11%20205525.png?raw=true)




### 112A. Petya and Strings

implementation, strings, 1000, http://codeforces.com/problemset/problem/112/A



˼·������ĿҪ����ɣ�ȫ��ת���ɴ�/Сд���бȽ�



##### ����

```python
string1 = input().strip()
string2 = input().strip()

string1_lower = string1.lower()
string2_lower = string2.lower()

if string1_lower < string2_lower:
    print(-1)
elif string1_lower > string2_lower:
    print(1)
else:
    print(0)
```



�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==

![q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-11%20205928.png?raw=true)




### 231A. Team

bruteforce, greedy, 800, http://codeforces.com/problemset/problem/231/A



˼·����ȡ�����㣬�ܽᣬ����



##### ����

```python
n = int(input())  
sol = 0

for i in range(n):
    arr = list(map(int, input().split())) 
    count = sum(arr)  
    if count >= 2: 
        sol += 1

print(sol)
```



�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==

![q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-11%20214454.png?raw=true)




## 2. ѧϰ�ܽ���ջ�
�⼸������ʵ������ѧPython����������������һ��ʼ����ϰ�ߣ�������C����дһ�飬Ȼ��ת����Python�����ǽ�����Ҳ�ܰ�Python�﷨���գ�ϣ��δ�����ܽ�Python�õø�˳�֣����⣬�ҵ�˼���ٶȻ����е�̫����̫ϰ��������һ�飬����̫��ʱ�䣬���ֺ�����´ο���ֱ�Ӷ���д�����Կ����У�Ҳ��һ�����ݣ�����죡