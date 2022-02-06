import requests
import bs4
import time
from selenium import webdriver
def add_https(string):
    if string.startswith('https://'):
        return string
    else:
        return 'https://'+string
def printwebsites(cishu):
    i=0
    while i<cishu:
        try:
            r=requests.get(add_https(input('please enter your url')))
            content = bs4.BeautifulSoup(r.content.decode('utf-8'), 'lxml')
            element = content.find()
            print(element)
            i+=1
        except:
            print('please enter again')

def baidu_search():
    try:
        content = input('enter what you want to search')
        driver = webdriver.Edge()
        driver.get('https://www.baidu.com')
        driver.find_element_by_xpath('//*[@id="kw"]').send_keys(content)
        driver.find_element_by_xpath('//*[@id="su"]').click()
        input()
    except:
        baidu_search()

printwebsites(3)
baidu_search()
