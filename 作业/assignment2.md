## 1. 题目

### 263A. Beautiful Matrix

https://codeforces.com/problemset/problem/263/A



思路：
共5行，1行1行输入，如果该行有1，找与中心的距离，即行差和列差的绝对值之和


##### 代码
Python:
```python
for i in range(5):
   s=input().split()
   if '1' in s:
       print(abs(i-2)+abs(s.index("1")-2)) #用.index 找出某个数的索引值（列）
       break

```

C:
```c
#include <stdio.h>

int main() {
    int arr[5][5], a, b;

    // 输入5x5矩阵
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            scanf("%d", &arr[i][j]);
            if (arr[i][j] == 1) {
                a = i; // 记录1所在的行号
                b = j; // 记录1所在的列号
            }
        }
    }

    // 计算将1移动到(2,2)所需的步数
    int steps = abs(a - 2) + abs(b - 2);

    // 输出所需的步数
    printf("%d\n", steps);

    return 0;
}
```


代码运行截图 ==（至少包含有"Accepted"）==
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20152248.png?raw=true)




### 1328A. Divisibility Problem

https://codeforces.com/problemset/problem/1328/A



思路：看两数能不能整除，不能a就+1，此时计数也+1，直到a能被B整除（坑）

以上思路亲试会runtimeerror，究其原因是在数字非常大的时候会导致过多的迭代，
所以比较好的方式是直接计算a距离能被b整除还差多少数



##### 代码
Python：
```python
n=int(input())
for _ in range(n):
	a,b=map(int,input().split())
	rem=a%b
	if rem==0:
		print(0)
	else:
		print(b-rem) #rem+(b-rem)=b就能%b

```

C：
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

代码运行截图 ==（至少包含有"Accepted"）==
![q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20212108.png?raw=true)




### 427A. Police Recruits

https://codeforces.com/problemset/problem/427/A



思路:一个一个数字遍历，如果先遇到-1，就罪+1，如果遇到警察，警察+1，那么之后遇到-1就可以抵消（警察-1）



##### 代码

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
    scanf("%d", &n);  // 输入事件数量
    int arr[10000];

    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) {
            // 新增警察
            police += arr[i];
        }
        else {
            // 遇到犯罪事件
            if (police > 0) {
                // 如果有警察，处理犯罪
                police--;
            }
            else {
                // 没有警察可用，记录未处理的犯罪
                crime++;
            }
        }
    }

    printf("%d\n", crime);  // 输出未处理的犯罪事件数

    return 0;
}
```


代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20212501.png?raw=true)




### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：
建立一个book数组以作标记，数组长度即为L长度，把区域内的数字对应的book位置标为1，这样即便重复出现也??，然后只需要计算book数组内0的部分


##### 代码

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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20214535.png?raw=true)




### sy60: 水仙花数II

https://sunnywhy.com/sfbj/3/1/60



思路：这题麻烦在输出，如果直接输出数组不满足题目要求，也不能join(ans),要变成str才输出



##### 代码

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

c:(更麻烦）
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

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20220345.png?raw=true)




### 01922: Ride to School

http://cs101.openjudge.cn/practice/01922/



思路：看了答案才有一定的思路，惭愧，把问题想得太复杂，实质上是最终跟着哪个骑手，看他到学校的时间+等他出发的时间就可以



##### 代码

```python
import math

while True:
    n = int(input())
    if n == 0:
        break

    fasttime = float("inf") #inf无穷，需用float
    for _ in range(n):
        speed, time = map(int, input().split())
        if time < 0: #快过他，省略
            continue
        arrivetime = math.ceil((4.5*3600) / speed + time) 
        fasttime = min(fasttime, arrivetime) #更新更快速度所花时间（取min）

    print(fasttime)

```

c里可以这么写：
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


代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-09-24%20232642.png?raw=true)




## 2. 学习总结和收获

距离第一次作业约莫两周，老实说刚开学有点懵，就有点懈怠了/(ㄒoㄒ)/~~，现在每日选做还在9月初的题，而且之前都是用C来做，不过听了学长姐的建议，决定还是好好学Python，所以又把之前没用python解的题再解了一遍，做了30+的题，确实对掌握Python语法很有帮助，自己也会在每次解题遇到新的语法时（问了GPT后）做记录，相信近日就能把握基础的语
法，不过还需要锻炼的还是自己的算法能力，每次看到新花样的题还是会懵，看了解答才惊觉不过如此，那些看似复杂
的情景题其实就是一开始学的基础语法的应用而已??。

希望能在国庆后熟练掌握python之余，还开始阅读算法相关的书/笔记，期许自己能独立解决1000+的题（不再依靠GPT）！\(￣︶￣*\))