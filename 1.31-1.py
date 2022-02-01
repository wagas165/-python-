import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.utils import formataddr
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
file=read_csvorxls()
my_sender = 'xxx'
my_pass = 'xxx'
dictionary = file.to_dict()
del dictionary['邮箱']
my_user=file['邮箱'].to_dict()
message={r:'' for r in range(0,len(my_user))}
for i,j in dictionary.items():
    for p in range(0,len(message)):
        message[p]+=str(f'{i}:{j[p]};')
def mail():
    ret = True
    try:
        for i,j in message.items():
            msg = MIMEText(j, 'plain', 'utf-8')
            msg['From'] = formataddr(['wagas165', my_sender])
            msg['To'] = formataddr(['',my_user[i]])
            msg['Subject'] = "阿巴巴巴"

            server = smtplib.SMTP_SSL("smtp.163.com")
            server.login(my_sender, my_pass)
            server.sendmail(my_sender,my_user[i], msg.as_string())
            server.quit()
    except Exception:
        ret = False
    return ret
ret = mail()
if ret:
    print("邮件发送成功")
else:
    print("邮件发送失败")