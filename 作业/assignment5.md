# Assignment #5: Greedy���Implementation

Updated 1939 GMT+8 Oct 21, 2024

2024 fall, Complied by <mark>��PȨ Ԫ��</mark>



**˵����**

1�����ÿ����Ŀ����˼·����ѡ����Դ��Python, ����C++���Ѿ���Codeforces/Openjudge��AC������ͼ������Accepted������д��������ҵģ���У��Ƽ�ʹ�� typora https://typoraio.cn ��������word����AC ����û��AC���������ÿ����Ŀ���»���ʱ�䡣

3���ύʱ�����ύpdf�ļ����ٰ�md����doc�ļ��ϴ����Ҳࡰ��ҵ���ۡ���Canvas��Ҫ��ͬѧ����ͷ���ύ�ļ���pdf��"��ҵ����"�����ϴ���md����doc������

4����������ڽ�ֹǰ�ύ��ҵ����д��ԭ��



## 1. ��Ŀ

### 04148: ��������

brute force, http://cs101.openjudge.cn/practice/04148

˼·��

˼·���ǴӸ�����������һ�쿪ʼ��ͬʱ����n-p��23�ı�����n����pһ�����ڣ�������߷壩��n-e��28�ı�������n-i��33�ı���,�ҵ��˾ͷ���n-d��Ϊ�𰸣��˷�����ö�٣��е��ʱ

��ʱԼ20����

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20153101.png?raw=true)




### 18211: ��������

greedy, two pointers, http://cs101.openjudge.cn/practice/18211

˼·��

���˺ܳ�ʱ���������ž�������ú󣬾ʹ���С�������������С��Ҳ����Ҳ����cnt=0���Ǿ�break������֮��ʣ��Ǯ���������򣬾������ģ��ٻ��������򣬲�֪���ǲ��Ǿ��Ƕ��ַ���Ȼ��Ƚ��鷳�ľ����������֮����������Ǵ�ģ����Լ���һ��max��������ÿ�����ʱ��cnt�����ֵ

��ʱ��33min

���룺

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



�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20162148.png?raw=true)




### 21554: �Ŷ���ʵ��

greedy, http://cs101.openjudge.cn/practice/21554

˼·��

���⵹�ǲ��ѣ�Ҫ�뵽��tuple�Ϳ����ˣ����ҴӲ������ݾͿ��Կ������з�ʽ���Ǹ���ʵ��ʱ�����̣�Ȼ��ȴ�ʱ����΢�ñ�ֽ�����¾Ϳ��Եó����㷽��������һλ��ʵ��ʱ���Ǻ��ÿ���˶�Ҫ�ȵģ�*(n-1),�Դ�����

��ʱ 20min����֪����ʱ�仨�Ķ��ˣ�

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20210252.png?raw=true)



### 01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/

˼·��

����һ���͵����ֵ��ˣ�Ȼ���ȴ�haab������������������������*365�죬�����·ݣ�1����20�죩+������Ȼ��ת�����������Ǹ�number��ÿ13�ظ�һ�Σ��м�Ĵ���ÿ20�ظ�

��ʱ 22min


���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-22%20232535.png?raw=true)




### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C

˼·��

�����Һ���CF���ܹ�͸���������������޸��Ҵ����©������Ҫ�ǵ�һ�������һ����ʵ��һ���ܿ��ģ�Ȼ����һ��Ҫע��ĵط������������ĳ���������ҵ�����ô���ж��¿����ܲ������󵹵�ʱ��prv����ָ�ľ�����һ������λ��Ӧ�ð����������ľ���

��ʱ30+min

���룺

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



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-23%20002606.png?raw=true)




### 01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/

˼·��

��һ�⻨�˺ܳ�ʱ�䣬��һֱRE�����ǿ������⣬Ȼ���о����Ǹ�y���ܴ���d������һ��ʼ���Ե���

����˼·�������״�ֻ�ܷ��ں������ϣ������״���뵺��Զ�ľ��룬��Ȼ�ǳ�һ��ֱ��������б�߷���ľ��룬���Բ���d^2-y^2����ȷ��һ�������ܱ��״��⵽���״�������õ����䣬Ȼ���ж�ÿ����������䣨�Ҳ��֣��Ƿ����ϸ���⵺������䣨�󲿷֣�������ɹ���ͬ���״����ͬ�״�

���룺

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
        islands.sort(key=lambda x:x[1]) #���Ҳ�������
        sta=float('-inf')
        for i in range(n):
            if sta<islands[i][0]:
                cnt+=1
                sta=islands[i][1]
    print(f"Case {ctt}: {cnt}")
    input()
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-23%20235825.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

�����ҵһ���ڻ������Ը㶨ǰ5�⣬��Ŀ���ܺã���ϰ������̰�ģ��ֵ�ȣ�Ҳ�ܿ����˵���ѧ�߼���������6����Ҫ��Ҫ���ܳ�ʱ�䲻���Դ���������ĵ�AC����Ϊ�����æ��ÿ��ѡ��û��ô������ʦ�Ŀμ�Ҳ����Ҫʱ���������Һ�����ҵ������������ȥ����ͬʱ��ѧ���������ҿ���Ҫ�ϵ��ܳٲ���ÿ��ѡ��������Ӧ����

��ε���ҵ�и����ѵľ���Ҫ����ʵ����ѧ���������ܽ�һ�����⣬���������㷨Ҳ���ѻش�

��Ӧ�������б�ȫ��׷�ϣ���ҵ�ѶȷŴ��Ҿ�����ʵ���õģ�����ʹ�ҡ���������ѧϰ����Ƚ�ǰ���Σ���ε�������ѧϰ��

<mark>�����ҵ��Ŀ�򵥣��з������ϰ��Ŀ�����磺OJ���Ƹ�2024fallÿ��ѡ������CF��LeetCode����ȵ���վ��Ŀ��</mark>