# Assignment #4: T-primes + ̰��


2024 fall, Complied by <mark>��PȨ Ԫ��</mark>



**˵����**

1�����ÿ����Ŀ����˼·����ѡ����Դ��Python, ����C++���Ѿ���Codeforces/Openjudge��AC������ͼ������Accepted������д��������ҵģ���У��Ƽ�ʹ�� typora https://typoraio.cn ��������word����AC ����û��AC���������ÿ����Ŀ���»���ʱ�䡣

3���ύʱ�����ύpdf�ļ����ٰ�md����doc�ļ��ϴ����Ҳࡰ��ҵ���ۡ���Canvas��Ҫ��ͬѧ����ͷ���ύ�ļ���pdf��"��ҵ����"�����ϴ���md����doc������

4����������ڽ�ֹǰ�ύ��ҵ����д��ԭ��



## 1. ��Ŀ

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



˼·��
Ҫ��׬���㣬�Ǿ������ģ��������򣨿���Ҳ���Բ�����Ȼ��Ѹ����ļ������ͺ��ˣ���ʱ7���ӣ�



����

```python
# 
n,m=map(int,input().split())
arr_1=list(map(int,input().split()))
arr_1=sorted(arr_1)
earn=0
for i in range(m):
    if arr_1[i]>0:
        break
    earn-=arr_1[i]
print(abs(earn))
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20171045.png?raw=true)




### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A

˼·��
Ҫ׬���ʹӴ��ֵ��ʼ����һ���õ�ǡ�ó���ʣ��ľ�ͣ�֣���ʱ5��04�룩


����

```python
n=int(input())
arr_1=list(map(int,input().split()))
c=sum(arr_1)
b=0
coin=0
arr_1=sorted(arr_1,reverse=True)
for i in arr_1:
    b+=i
    coin+=1
    if b>c//2:
        break
print(coin)
```



�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20172124.png?raw=true)




### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B

˼·��

���˴��ʱ�䳢�Զ�����Ŀ�����Ƿ���ȥ���������ݸ��ã���������˵����һ��n*n�������ÿ��/ÿ�ж���һ��chips��amount�����������������һ����������С����ȥ��ÿ�е��� ����������С��ȥ��ÿ�е������Ƚϼ��ɵ�

����ʱ15��43�룩

����

```python
t=int(input())
for _ in range(t):
    n=int(input())
    a=list(map(int,input().split()))
    b=list(map(int,input().split()))
    min_a=min(a)
    min_b=min(b)
    asw_a=sum([min_a+i for i in b])
    asw_b=sum([min_b+i for i in a])
    print(min(asw_a,asw_b))
```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20174337.png?raw=true)




### 158B. Taxi

*special problem, greedy, implementation, 1100, https://codeforces.com/problemset/problem/158/B

˼·��

Ӧ��˵�����CF�ܿ����������Ҳ���һ������©��ȱ������д�úܷ������������뷨�����һ��������3��4��Ҫһ�������һ��2�ĳɶԾ�+1������ʣ�µ�һ��2�˵ľ�Ҫһ����Ȼ��1�˵ľͿ�3�˺�2���Ǳ��ܲ�������

(��ʱ28���ӣ��ȴ���������ʱ)

����

```python
import math
n=int(input())
arr_1=list(map(int,input().split()))
car=0
sum2=0
sum3=0
sum1=0
for i in arr_1:
    if i==4:
        car+=1
    elif i==3:
        car+=1
        sum3+=1
    elif i==2:
        sum2+=i
        if sum2==4:
            car+=1
            sum2-=4
    else:
        sum1+=i
if sum2>0:
    car+=1
sum31=sum3+sum2-sum1
if sum31<0:
    while abs(sum31)>4:
        sum31+=4
        car+=1
    if sum31<0:
        car+=1
print(car)
```

�Ż������£�����AC��
```python
import math
n = int(input())
arr_1 = list(map(int, input().split()))

car = 0
sum1 = sum2 = sum3 = 0

for i in arr_1:
    if i == 4:
        car += 1  # ÿ��4���˶���Ҫһ����
    elif i == 3:
        sum3 += 1  # ��¼3������
    elif i == 2:
        sum2 += 1  # ��¼2������
    else:
        sum1 += 1  # ��¼1������

# �ȴ���3���˺�1����
min_3_and_1 = min(sum3, sum1)
car += sum3  # 3����������Ҫһ����
sum1 -= min_3_and_1  # ��ȥ��3��Ե�1

# ����2���ˣ�ÿ����2����ƴ��һ����
car += sum2 // 2
if sum2 % 2:  # �������һ��ʣ���2
    car += 1
    sum1 = max(0, sum1 - 2)  # �����ʣ���1����ȥ���2��

# ʣ�µ�1���ˣ�ÿ4��1����ƴ��һ����
car += math.ceil(sum1 / 4 ) #����ȡ��

print(car)
```


�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20182848.png?raw=true)




### *230B. T-primes��ѡ����

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B

˼·��

������ʵ�ܺ���⣬ֻҪһ��������������������Tprime���ҳ�ȥ1�ͱ�����һ��������ΪTprime�����ƽ�����ұ�������������ؼ������ж��������ֻ�TLE

�������˰���ɸ�������ҵ�һ����Χ�����������Ա㲻��Ҫÿ������ȥ��һ���ж������ĺ���������ɸ������˵�����������ı�����������������������������Ѿ����������ı�����ô�Ͳ���Ҫ������ı����ˣ��ݴ˾Ϳ��Ա�������ǺͲ�����������

����ʱԼ15���ӣ�

����

```python
import math
def isprime(limit):
    is_prime=[True]*(limit+1)
    is_prime[0]=is_prime[1]=False
    for i in range(2,int(math.sqrt(limit))+1):
        if is_prime[i]:
            for j in range(i*i,limit+1,i):
                is_prime[j]=False
    return is_prime


max_lim=10**6
prime_num=isprime(max_lim)

n=int(input())
arr=list(map(int,input().split()))

for i in arr:
    sqrt_1=math.isqrt(i)
    if sqrt_1*sqrt_1==i and prime_num[sqrt_1]:
        print('YES')
    else:
        print('NO')

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20185537.png?raw=true)




### *12559: �����С���� ��ѡ����

greedy, strings, sortings, http://cs101.openjudge.cn/practice/12559

˼·��

���������ֵ���Ƚϣ������ֱȽϣ�sorted�����Ϳ��Դ�ɣ���������Щ���ֲ�������������������еġ�11���͡�113������ô�ͽ����ֶ���������10���ֵĳ��ȣ��ظ�10�Σ����ٽ��бȽϼ���

��ȻaiҲ�����������cmp_to_key��ʽ�Զ���sorted������ʽ����������Ϊ���Ը��ӣ�

����ʱԼ24���ӣ�

����

```python
n=int(input())
arr_1=list(input().split())
arr_2=arr_1.copy()
max_num = ''.join(sorted(arr_1, key=lambda x: x*10, reverse=True))

min_num = ''.join(sorted(arr_2, key=lambda x: x*10))
print(max_num ,min_num)

```



�������н�ͼ <mark>�����ٰ�����"Accepted"��</mark>
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-15%20221323.png?raw=true)




## 2. ѧϰ�ܽ���ջ�
�����ҵ�����Ŀ����֮ǰÿ��ѡ�������ģ�������ǰû���������������ˣ����������ĵ�

taxi������Ȼ���˺ܶ�ʱ�䣬��������˼·ȥ���ģ����˸о���װ��������һ������ƣ�������Ϊ��˺ܿ�Ϳ����뵽3��4����һ������Ȼ���ٴ���2���˺�1���˵�����

Tprime�������Ű�֮ǰѧ�İ���ɸ������д�ˣ��󲿷ֻ���д���ˣ���Ȼ����һ�㣩���о������ѧ��

���һ��Ļ�������ai��֪���ظ�����ٱȽ�

֮ǰһֱˢһ����һ�⣬һֱ��Ϊûʲôѧϰ��չ�������ظ�����ȥ�ŷ�����ʵѧ������ģ��־����Լ����ˣ�����ˢ�⣡����δ��ѡ�������ꡣ����

Ȼ����������ҵ���ú�ʱ�����ţ��Ͼ����⻷�����̫���ʡ����´��Կ������ϼ�ʱ��

<mark>�����ҵ��Ŀ�򵥣��з������ϰ��Ŀ�����磺OJ���Ƹ�2024fallÿ��ѡ������CF��LeetCode����ȵ���վ��Ŀ��</mark>