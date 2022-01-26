import pandas as pd
import re
def read_csvorxls():
    filepath=input('please enter your filepath')
    #先判断后缀名
    if filepath.endswith('.csv')==True:
        return pd.DataFrame(pd.read_csv(filepath))
    elif filepath.endswith('.xls')==True or filepath.endswith('.xlsx')==True:
        return pd.DataFrame(pd.read_excel(filepath))
    else:
        print('plese enter the right name(example:"xxxx.xls")')
        read_csvorxls()
def modify(file,row,col,newvalue):#指定位置赋新值
    file.iloc[row-1,col-1]=newvalue
    print(file)
def exchange(file,r1,c1,r2,c2):#交换位置
    mid=file.iloc[r1-1,c1-1]
    file.iloc[r1-1,c1-1]=file.iloc[r2-1,c2-1]
    file.iloc[r2-1,c2-1]=mid
    print(file)
def delete(file,r1,c1):#删除指定行，指定列
    file.drop(columns=str(file.columns[c1-1]),axis=1,inplace=True)
    file.drop([int(r1-1)],axis=0,inplace=True)
    print(file)
data=read_csvorxls()
print(data)
manipulation=input('please input your manipulation:')
list=[]
if manipulation.startswith('modify'):
    for i in re.sub('\D', '', manipulation):
        list.append(int(i))
    if len(list)==3:
        modify(data,list[0],list[1],list[2])
    else:
        print('error! has incorrect variables.')
elif manipulation.startswith('exchange'):
    for i in re.sub('\D', '', manipulation):
        list.append(int(i))
    if len(list)==4:
        exchange(data,list[0],list[1],list[2],list[3])
    else:
        print('error! has incorrect variables.')
elif manipulation.startswith('delete'):
    for i in re.sub('\D','',manipulation):
        list.append(int(i))
    if len(list)==2:
        delete(data,list[0],list[1])
    else:
        print('error! has incorrect variables.')
else:
    print('error! no such manipulation')


