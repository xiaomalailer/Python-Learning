# Assign #3: Oct Mock Exam暨选做题目满百

马凱权 2400094604 元培


**说明：**

Oct⽉考： AC6  ==3== 


## 1. 题目

### E28674:《黑神话：悟空》之加密

http://cs101.openjudge.cn/practice/28674/



思路：

思路备注在代码旁了，可能有其他更好的做法，因为此做法需了解字母在ASCII的位置（可以自己跑代码找出来），然后通过数算去找规律

更新：在群里看到不错的做法，用tuple把26个字母大小写组合，如('a','A')，用isupper函数判断，0小写，1大写

代码

```python
n = int(input())  
arr = input()  
while n > 26:  # 如果n大于26，将其缩小到26以内（因为字母表只有26个字母）
    n -= 26
for i in arr:  
    anw = 0  
    anw = ord(i) - n  # 将字符i的ASCII值向左移位n个位置
    if anw < 97 and ord(i) > 96:  # 如果计算后的值小于97（即小于'a'），并且i是小写字母
        anw = ord(i) - n + 26  # 回到字母表的末尾
    elif anw < 65 and ord(i) < 91:  # 如果计算后的值小于65（即小于'A'），并且i是大写字母
        anw = ord(i) - n + 26  # 回到字母表的末尾
    print(chr(anw), end='')  #end=''确保输出在同一行
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q1](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-10%20220401.png?raw=true)




### E28691: 字符串中的整数求和

http://cs101.openjudge.cn/practice/28691/



思路：
只选取输入的前两个即为数字，来做简单加法


代码

```python
a,b=input().split()
a_1=a[0:2]
b_1=b[0:2]
print(int(a_1)+int(b_1))
```



代码运行截图 ==（至少包含有"Accepted"）==
![Q2](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-10%20220850.png?raw=true)




### M28664: 验证身份证号

http://cs101.openjudge.cn/practice/28664/



思路：
此代码没有AC，一直RE，直到月考结束后做了小改动才AC，难绷

思路为建立两个数列，一个id存放系数，一个psw存放最后前十七个数*系数之和再除以11后应得的尾数
然后就照着题目指示相乘相加，然后比对尾数和总和%11对应的数就可以了

代码

```python
n = int(input())
id = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
psw = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
for _ in range(n):
    a = input()

    sum = 0
    for i in range(17):
        sum += int(a[i]) * id[i]
    sum %= 11

    if a[17] == psw[sum]:
        print("YES")
    else:
        print("NO")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![Q3](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-10%20221801.png?raw=true)




### M28678: 角谷猜想

http://cs101.openjudge.cn/practice/28678/



思路：
按题目要求就可完成，使用while else处理


代码

```python
n=int(input())
while n!=1:
    if n%2!=0:
        print(f"{n}*3+1={n*3+1}")
        n=n*3+1
    else:
        print(f"{n}/2={n//2}")
        n=n//2
else:
    print('End')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![Q4](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-10%20223852.png?raw=true)



### M28700: 罗马数字与整数的转换

http://cs101.openjudge.cn/practice/28700/



思路：

首先先判断是整数还是罗马数字（非整数），
for 罗马数字：就判断那些特定的情况，非特定情况就从dict里找
for 整数：由于给定条件不会到五位数，那就只需要先看能有几个1000，



##### 代码

```python
arr = input()
rome_1 = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}

sum = 0
i = 0
if arr[i] not in '1234567890':
    while i < len(arr):
        if i < len(arr) - 1:
            if arr[i] == 'I' and arr[i + 1] == 'V':  # IV
                sum += 4
                i += 2
                continue
            elif arr[i] == 'I' and arr[i + 1] == 'X':  # IX
                sum += 9
                i += 2
                continue
            elif arr[i] == 'X' and arr[i + 1] == 'L':  # XL
                sum += 40
                i += 2
                continue
            elif arr[i] == 'X' and arr[i + 1] == 'C':  # XC
                sum += 90
                i += 2
                continue
            elif arr[i] == 'C' and arr[i + 1] == 'D':  # CD
                sum += 400
                i += 2
                continue
            elif arr[i] == 'C' and arr[i + 1] == 'M':  # CM
                sum += 900
                i += 2
                continue
        sum += rome_1.get(arr[i], 0)
        i += 1

    print(sum)

else:
    num=int(arr)
    rome_2=[]
    while num >= 1000:
        rome_2.append('M')
        num -= 1000

    if num >= 900:
        rome_2.append('CM')
        num -= 900
    elif num >= 500:
        rome_2.append('D')
        num -= 500
    elif num >= 400:
        rome_2.append('CD')
        num -= 400
    while num >= 100:
        rome_2.append('C')
        num -= 100

    if num >= 90:
        rome_2.append('XC')
        num -= 90
    elif num >= 50:
        rome_2.append('L')
        num -= 50
    elif num >= 40:
        rome_2.append('XL')
        num -= 40
    while num >= 10:
        rome_2.append('X')
        num -= 10

    if num == 9:
        rome_2.append('IX')
        num -= 9
    elif num >= 5:
        rome_2.append('V')
        num -= 5
    elif num == 4:
        rome_2.append('IV')
        num -= 4
    while num >= 1:
        rome_2.append('I')
        num -= 1

    print(''.join(rome_2))

```

之后让gpt帮我优化了，发现函数和字典真的好好用
``` python
def roman_to_int(roman):
    rome_1 = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    special_cases = {'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900}
    
    i, total = 0, 0
    while i < len(roman):
        if i < len(roman) - 1 and roman[i:i+2] in special_cases:  # 处理特殊情况（如IV, IX等）
            total += special_cases[roman[i:i+2]]
            i += 2
        else:
            total += rome_1[roman[i]]
            i += 1
    return total

def int_to_roman(num):
    val_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), 
               (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), 
               (5, 'V'), (4, 'IV'), (1, 'I')]
    
    roman = []
    for val, symbol in val_map:
        while num >= val:
            roman.append(symbol)
            num -= val
    return ''.join(roman)

arr = input()

if arr.isdigit():  # 如果输入是数字，转换为罗马数字
    num = int(arr)
    print(int_to_roman(num))
else:  # 如果输入是罗马数字，转换为整数
    print(roman_to_int(arr))

```

代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![Q5](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-11%20000308.png?raw=true)




### *T25353: 排队 （选做）

http://cs101.openjudge.cn/practice/25353/



思路：

其实思考线性树到底怎样运作就花了好多时间，才大致理解了每个代码的用途，让我写出来现在大概还不能

以下代码都是抄自解答，下面将以我的理解方式试着叙述：

以   D=3, h=[7,7,3,6,2] 为例，

current_layers主要是用来求一个同学前后的同学的最高层数，+1就是要排在最高层数的下一层

比如一开始h1=7，检查前面有无大山，也就是身高差值大于D的，对身高7来说就是[1,3]这个范围，因为[1,3]范围无层数数据，返回结果是0，再看大于7部分有无可以交换的，返回结果是0，
因此对于h1=7属于第一层[[],[7]],对于h2=7同理，所以目前有[[],[7,7]]

接着是h3=3，检查前面部分就是[1,-1],显然返回结果是0，后面部分是[7,7]，有且仅有一个层数，也就是7层数为1，所以h3=3的层数为2，有[[],[7,7],[3]]

接着是h4=6，检查前面部分[1,2],返回0，后面部分[10,7]无效，返回0，所以6的层数也是1，有[[],[7,7,6],[3]]

最后是h5=2，检查前面部分[1,-2]返回0,后面部分[5,7]，有6，7的层数都是1，所以2的层数是2，有[[],[7,7,6],[3,2]]

最后再sort下输出即可

此方法其实就是将数据分层，根据前面有的“大山”的层数判断要放到第几层，然后每每为一个数据分层了，就要modify来更新节点的值



代码

```python
解释每一步：# 使用class写一棵线段树
class SegmentTreeNode:
    def __init__(self, start, end, val):
        self.start = start
        self.end = end
        self.val = val
        self.lson = None
        self.rson = None


class SegmentTree:
    def __init__(self, start, end):
        self.root = SegmentTreeNode(start, end, 0)
        self.start = start
        self.end = end

    # 返回修改后的节点的对象。注意对象是浅拷贝，里面存储了一些指针。
    # 改变了子对象的指针之后，需要重新赋值给父对象，不然子对象的子对象就会被释放
    def modify(self, node, pos, value, start=None, end=None):
        if (start is None):
            start, end = self.start, self.end
        if (node is None):
            node = SegmentTreeNode(start, end, value)
        if (start == end):
            node.val = max(node.val, value)
            return node
        mid = (node.start + node.end) // 2
        if pos <= mid:
            node.lson = self.modify(node.lson, pos, value, start, mid)
        else:
            node.rson = self.modify(node.rson, pos, value, mid + 1, end)
        node.val = max(node.val, value)
        return node

    def query(self, node, qstart, qend):
        if ((node is None) or qstart > node.end or qend < node.start):
            return 0
        if (qstart <= node.start and node.end <= qend):
            return node.val
        ret = max(self.query(node.lson, qstart, qend), self.query(node.rson, qstart, qend))
        return ret


# 主程序非常短
N, D = map(int, input().split())
h = [int(input()) for _ in range(N)]
# 求出每个点的层数，用一个线段树维护值域上的层数，这样就可以求出挡在前面的大山的最大的层数了
MAXH = max(h)
layers = SegmentTree(1, MAXH)
members = [[]]  # 存储每一层里面的元素
for hi in h:
    current_layer = max(layers.query(layers.root, 1, hi - D - 1), layers.query(layers.root, hi + D + 1, MAXH)) + 1
    if (current_layer >= len(members)):
        members.append([])
    members[current_layer].append(hi)
    layers.modify(layers.root, hi, current_layer)

# 直接一层层排序、输出即可
for layer in members:
    for _ in sorted(layer):
        print(_)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![Q6](https://github.com/xiaomalailer/img/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-10-13%20143542.png?raw=true)




## 2. 学习总结和收获
月考自认为没发挥好，只AC了三题，主要是身份证号那题不知怎么地因为多加了一些函数，就一直RE
,结果月考完直接做个小改动的就AC了，然后前面浪费太多时间了，比平时做选做题还花了更多时间去做easy题，罗马数字那题当时没时间看，但是之后去做发现思路不难，只是可能写得有点长，排队就不说了，搞不明白，最后只能求助于解答，当然自己从周四开始到周日都在思考如何解，但都是O(N^2)，所以那么迟交上作业

另外国庆假期其实没什么碰到题，是在假期结束前两天狂刷题，终于把进度赶到10月初的题，Tprime的题思路是对的但是没学过埃氏或欧拉筛法所以一直超时，应该要去看些算法教学的书本等来学学

目前来看8，900的题估计问题不大，最多是解题速度要提上来，1000+的题感觉还不是很能把握，估计是自己算法基础太薄弱，而且感觉自己写的代码每次都太冗长，这次月考才发现优化代码避免超时的重要性

给自己的目标大概就是下次月考至少AC4题，最好是5题，tough的题就看运气了，当然希望这个月能抓紧时间学习算法，争取至少能和难题斗一斗

==如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。==