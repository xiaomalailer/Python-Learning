### 模板
link：

思路：

####代码
```python

```
学习总结：


### 02733: 判断闰年
link:http://cs101.openjudge.cn/practice/02733/

思路：
能被3200整除的非闰年，能被100整除但是不能被400的整除的也非闰年，余下能被4整除的为闰年，其他为平年

##### 代码

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
学习总结：


### 02750 鸡兔同笼
link:http://cs101.openjudge.cn/2024fallroutine/02750/

思路：
如果脚的总数为4的倍数，那么最少就是全部兔子
如果脚的总数不为4的倍数，那么最少就是全兔子+1鸡

##### 代码

```python
a = int(input()) #输入正整数a
if a%4 == 0:  #如果脚的总数为4的倍数，那么最少就是全部兔子
    print(int(a/4), int(a/2))
elif a%2 == 0: #如果脚的总数不为4的倍数，那么最少就是全兔子+1鸡
    print(int((a+2)/4), int(a/2))
else:
    print(0, 0)
```
学习总结：输入方式 input( ) 要加int如果输入数字


