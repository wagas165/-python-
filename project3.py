import time,random
start=0
time_1 = int(time.time())   #初始时间
init = time_1 % 24   #以对应小时数开始
def timer():
    global start,time_1,init
    if start==0:
        start+=1
        return init%24
    else:
        start+=1
        time_2=int(time.time())
        add=((time_2-time_1)//5)%24 #加几个小时
        return (add+init)%24
def command():
    comman=input('你想：')
    if comman=='bye':
        f=open("C:\\Users\\31626\\Desktop\\tommy.txt",'w')
        f.write(f'{tommy.status_1}\n{tommy.happy}\n{tommy.hungry}\n{tommy.healthy}')
        print('记得来找我！Bye.....')
    elif comman=='walk' or comman=='play' or comman=='feed' or comman=='seedoctor' or comman=='letalone' or comman=='status':
        eval('tommy.'+comman)()
    else:
        print('我不懂你在说什么')
        command()
class Cat():
    def __init__(self):
        self.happy=random.randint(0,100)
        self.hungry=random.randint(0,100)
        self.healthy=random.randint(0,100)
        if timer()>=8:
            self.status_1='醒着但很无聊'
        else:
            self.status_1='在睡觉'
        print('我的名字叫Tommy，一只可爱的猫咪....')
        print('你可以跟我一起散步，玩耍，你也需要给我好吃的东西，带我去看病，也可以让我发呆....')
        print('Commands:\n1.walk:散步\n2.play:玩耍\n3.feed:喂我\n4.seedoctor:看医生\n5.letalone:让我独自一人\n6.status:查看我的状态\n7.bye:不想看到我\n')
    def sleep(self):
        if timer()>=8:
            self.status_1='醒着但很无聊'
        else:
            self.status_1='在睡觉'
    def extra(self):
        if self.status_1=='在睡觉':
            command2=input('你确认要吵醒我吗？我在睡觉，你要是坚持吵醒我，我会不高兴的！（y表示是/其他表示不是）')
            if command2=='y':
                self.happy-=4
                return True
        else:
            return True
    def walk(self):
        self.sleep()
        if self.extra()==True:
            self.status_1='在散步'
            print(f'我{self.status_1}......')
            if self.status_1=='在散步':
                self.hungry += 2
                self.healthy += 1
            self.other()
            time.sleep(5)
        command()
    def play(self):
        self.sleep()
        if self.extra()==True:
            self.status_1='在玩耍'
            print(f'我{self.status_1}......')
            if self.status_1=='在玩耍':
                self.hungry += 3
                self.happy += 1
            self.other()
            time.sleep(5)
        command()
    def feed(self):
        self.sleep()
        if self.extra()==True:
            self.status_1='在吃饭'
            print(f'我{self.status_1}......')
            if self.status_1=='在吃饭':
                self.hungry -= 3
            self.other()
            time.sleep(5)
        command()
    def seedoctor(self):
        self.sleep()
        if self.extra()==True:
            self.status_1='在看医生'
            print(f'我{self.status_1}......')
            if self.status_1=='在看医生':
                self.healthy += 4
            self.other()
            time.sleep(5)
        command()
    def letalone(self):
        self.sleep()
        if self.extra() == True:
            self.status_1='醒着但很无聊'
            print(f'我{self.status_1}......')
            if self.status_1=='醒着但很无聊':
                self.hungry += 2
                self.happy -= 1
            self.other()
            time.sleep(5)
        command()
    def other(self):
        if self.hungry>80 or self.hungry<20:
            self.healthy-=2
            time.sleep(5)
        if self.happy<20:
            self.healthy-=1
            time.sleep(5)
    def status(self):
        self.sleep()
        print(f'当前时间：{timer()}点\n我当前的状态：我{self.status_1}......')
        def print_status(a):
            if a==100:
                return a
            elif a>=10:
                return f'0{a}'
            else:
                return f'00{a}'
        print('Happiness:'+'Sad'+("*"*int(self.happy//2)+"-"*(50-int(self.happy//2)))+f'Happy({print_status(self.happy)})')
        print('Hungry:'+'Full'+("*"*int(self.hungry//2)+"-"*(50-int(self.hungry//2)))+f'Hungry({print_status(self.hungry)})')
        print('Health:'+'Sick'+("*"*int(self.healthy//2)+"-"*(50-int(self.healthy//2)))+f'Healthy({print_status(self.healthy)})')
        command()
try:
    f=open("C:\\Users\\31626\\Desktop\\tommy.txt")
    line=f.readlines()
    tommy=Cat()
    tommy.status_1=line[0]
    tommy.happy=int(line[1])
    tommy.hungry=int(line[2])
    tommy.healthy=int(line[3])
except:
    tommy=Cat()

command()




