## 1. ��Ŀ

### 263A. Beautiful Matrix

https://codeforces.com/problemset/problem/263/A



˼·��
��5�У�1��1�����룬���������1���������ĵľ��룬���в���в�ľ���ֵ֮��


##### ����
Python:
```python
for i in range(5):
   s=input().split()
   if '1' in s:
       print(abs(i-2)+abs(s.index("1")-2)) #��.index �ҳ�ĳ����������ֵ���У�
       break

```

C:
```c
#include <stdio.h>

int main() {
    int arr[5][5], a, b;

    // ����5x5����
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            scanf("%d", &arr[i][j]);
            if (arr[i][j] == 1) {
                a = i; // ��¼1���ڵ��к�
                b = j; // ��¼1���ڵ��к�
            }
        }
    }

    // ���㽫1�ƶ���(2,2)����Ĳ���
    int steps = abs(a - 2) + abs(b - 2);

    // �������Ĳ���
    printf("%d\n", steps);

    return 0;
}
```


�������н�ͼ ==�����ٰ�����"Accepted"��==
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20152248.png?raw=true)




### 1328A. Divisibility Problem

https://codeforces.com/problemset/problem/1328/A



˼·���������ܲ�������������a��+1����ʱ����Ҳ+1��ֱ��a�ܱ�B�������ӣ�

����˼·���Ի�runtimeerror������ԭ���������ַǳ����ʱ��ᵼ�¹���ĵ�����
���ԱȽϺõķ�ʽ��ֱ�Ӽ���a�����ܱ�b�������������



##### ����
Python��
```python
n=int(input())
for _ in range(n):
	a,b=map(int,input().split())
	rem=a%b
	if rem==0:
		print(0)
	else:
		print(b-rem) #rem+(b-rem)=b����%b

```

C��
```c
#include <stdio.h>

int main() {
    long long int n, a, b;
    scanf("%lld", &n);  
    for (int i = 0; i < n; i++) {
        scanf("%lld %lld", &a, &b);  
        long long int rem = a % b;
        if (rem == 0) {
            printf("0\n");
        } else {
            printf("%lld\n", b - rem);
        }
    }
    return 0;
}
```

�������н�ͼ ==�����ٰ�����"Accepted"��==
![q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20212108.png?raw=true)




### 427A. Police Recruits

https://codeforces.com/problemset/problem/427/A



˼·:һ��һ�����ֱ��������������-1������+1������������죬����+1����ô֮������-1�Ϳ��Ե���������-1��



##### ����

Python:
```python
n = int(input())
a = list(map(int, input().split()))
cnt = 0
police = 0
for i in a:
    if i == -1 and police == 0:
        cnt += 1
        continue
    if i > 0:
        police += i
        continue
    police -= 1

print(cnt)
```

C:
```c
#include <stdio.h>

int main() {
    int n, crime = 0, police = 0;
    scanf("%d", &n);  // �����¼�����
    int arr[10000];

    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) {
            // ��������
            police += arr[i];
        }
        else {
            // ���������¼�
            if (police > 0) {
                // ����о��죬������
                police--;
            }
            else {
                // û�о�����ã���¼δ����ķ���
                crime++;
            }
        }
    }

    printf("%d\n", crime);  // ���δ����ķ����¼���

    return 0;
}
```


�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==
![q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20212501.png?raw=true)




### 02808: У�������

http://cs101.openjudge.cn/practice/02808/



˼·��
����һ��book����������ǣ����鳤�ȼ�ΪL���ȣ��������ڵ����ֶ�Ӧ��bookλ�ñ�Ϊ1�����������ظ�����Ҳ??��Ȼ��ֻ��Ҫ����book������0�Ĳ���


##### ����

```python
L,m=map(int,input().split())
book=(L+1)*[0]
cnt=0
for _ in range(m):
    a,b=map(int,input().split())
    for i in range(a,b+1):
        book[i]=1
for i in range(L+1):
    if book[i]==0:
        cnt+=1
print(cnt)
```



�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==
![q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20214535.png?raw=true)




### sy60: ˮ�ɻ���II

https://sunnywhy.com/sfbj/3/1/60



˼·�������鷳����������ֱ��������鲻������ĿҪ��Ҳ����join(ans),Ҫ���str�����



##### ����

```python
a,b=map(int,input().split())
cnt = 0
ans=[]
for i in range(a,b+1):
    x=i%10
    y=(i//10)%10
    z=i//100
    if i== x**3+y**3+z**3:
        ans.append(i)
        cnt+=1
if cnt==0:
    print("NO")
else:
    print(" ".join(map(str,ans)))

```

c:(���鷳��
```c
#include<stdio.h>

int main(){
    int a,b,cnt=0;
    int result[100];
    scanf("%d %d",&a,&b);
    for(int i=a;i<=b;i++){
        int x=i%10;
        int y=(i/10)%10;
        int z=i/100;
        if (i==(x*x*x+y*y*y+z*z*z)) result[cnt++]=i;
    }
    if(cnt==0)printf("NO");
    else {
        for(int i=0;i<cnt;i++){
            printf("%d",result[i]);
            if(i<cnt-1) printf(" ");
        }
    }
    return 0;
}
```

�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==
![q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20220345.png?raw=true)




### 01922: Ride to School

http://cs101.openjudge.cn/practice/01922/



˼·�����˴𰸲���һ����˼·�����������������̫���ӣ�ʵ���������ո����ĸ����֣�������ѧУ��ʱ��+����������ʱ��Ϳ���



##### ����

```python
import math

while True:
    n = int(input())
    if n == 0:
        break

    fasttime = float("inf") #inf�������float
    for _ in range(n):
        speed, time = map(int, input().split())
        if time < 0: #�������ʡ��
            continue
        arrivetime = math.ceil((4.5*3600) / speed + time) 
        fasttime = min(fasttime, arrivetime) #���¸����ٶ�����ʱ�䣨ȡmin��

    print(fasttime)

```

c�������ôд��
```c
#include <stdio.h>
#include <math.h>

int main() {
    int n;  
    while (1) {
        scanf("%d", &n);  
        if (n == 0) break; 

        double fasttime = INFINITY;  
        for (int i = 0; i < n; i++) {
            int speed, time;
            scanf("%d %d", &speed, &time);  
            if (time < 0) continue; 

            double arrivetime = ceil((4.5*3600.0 / speed) + time);
           
            if (arrivetime < fasttime) {
                fasttime = arrivetime;
            }
        }

        
        printf("%d\n", (int)fasttime);  
    }

    return 0;
}

```


�������н�ͼ ==��AC�����ͼ�����ٰ�����"Accepted"��==
![q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20232642.png?raw=true)




## 2. ѧϰ�ܽ���ջ�

�����һ����ҵԼĪ���ܣ���ʵ˵�տ�ѧ�е��£����е�и����/(��o��)/~~������ÿ��ѡ������9�³����⣬����֮ǰ������C��������������ѧ����Ľ��飬�������Ǻú�ѧPython�������ְ�֮ǰû��python������ٽ���һ�飬����30+���⣬ȷʵ������Python�﷨���а������Լ�Ҳ����ÿ�ν��������µ��﷨ʱ������GPT������¼�����Ž��վ��ܰ��ջ�������
������������Ҫ�����Ļ����Լ����㷨������ÿ�ο����»������⻹�ǻ��£����˽��ž���������ˣ���Щ���Ƹ���
���龰����ʵ����һ��ʼѧ�Ļ����﷨��Ӧ�ö���??��

ϣ�����ڹ������������python֮�࣬����ʼ�Ķ��㷨��ص���/�ʼǣ������Լ��ܶ������1000+���⣨��������GPT����\(�����*\))