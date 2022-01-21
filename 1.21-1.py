# for floor in range(1,7):
#     print(f'当前在{floor}层'.center(50,'*'))
#     for room in range(1,11):
#         if room <10:
#             print(f'现在在查{floor}0{room}室',end=' ')
#         else:
#             print(f'现在在查{floor}{room}室')
# def find_prime(number):
#     prime=[]
#     for i in range(2,number+1):
#         prime.append(i)
#         for p in range(2,round(i**(1/2)+1)):
#             if i % p ==0:
#                 prime.remove(i)
#                 break
#
#     print(f'{number}里的素数有：', end=' ')
#     for i in prime:
#         print(i,end=' ')
# find_prime(100)
def huiwen(string):   #判断是否为回文数
    a=[]
    for i in string:
        a.append(i)
    length=0
    if len(a)%2==0:
        length=len(a)/2
    else:
        length=(len(a)-1)/2
    for i in range(0,int(length)):
        if a[i]!=a[(len(a)-i-1)]:
            return False
    else:return True

def huiwen_last4(number):   #判断最后四位是否为回文数
    number%=10000
    if number>=1000:
        number_string=str(number)
    elif number>=100:
        number_string='0'+str(number)
    elif number>=10:
        number_string = '00' + str(number)
    else:
        number_string = '000' + str(number)
    if huiwen(number_string)==True:
        return True
    else:
        return False

def huiwen_last5(number):   #判断最后五位是否为回文数
    number%=100000
    if number>=10000:
        number_string=str(number)
    elif number>=1000:
        number_string='0'+str(number)
    elif number>=100:
        number_string = '00' + str(number)
    elif number>=10:
        number_string = '000' + str(number)
    else:
        number_string = '0000' + str(number)
    if huiwen(number_string)==True:
        return True
    else:
        return False
def huiwen_middle4(number): #判断中间四位是否为回文数
    number=int((number%100000)/10)
    if huiwen_last4(number)==True:
        return True
    else:
        return False
print('你的里程数是',end='')
for i in range(100000,1000000):
    if huiwen_last4(i)==True and huiwen_last5(i)==False and huiwen_last5(i+1)==True and huiwen_middle4(i+2)==True and huiwen(str(i+3))==True:
        print(i)


