# 显然，如果直接调用正则模块，立刻秒杀
# import re
# def isMatch(s,p):
#     if re.fullmatch(p,s) != None:
#         return True
#     else:
#         return False
def isMatch(s, p):
    list= [[False for i in range(len(p) + 1)] for i in range(len(s) + 1)]
    #记录s,p在到i，j位是否匹配
    list[0][0] = True
    for i in range(1, len(p) + 1):
        if p[i - 1] == '*':    #如果p出现‘*’,考虑第i-2位
            list[0][i] = list[0][i - 2]
    for i in range(1, len(list)):
        for j in range(1, len(list[0])):
            if p[j - 1] == '*':
                if p[j - 2] == s[i - 1] or p[j - 2] == ".": #j-2位匹配
                    list[i][j] = (list[i - 1][j] or list[i - 1][j - 2] or list[i][j - 2])
                else:
                    list[i][j] = list[i][j - 2]
            elif s[i - 1] != p[j - 1] and p[j - 1] != '.': #i-1,j-1不匹配
                list[i][j] = False
            else: #即没有‘*’且i-1，j-1匹配
                list[i][j] = list[i - 1][j - 1]
    return list[len(s)][len(p)]
print(isMatch('aa','a'))
print(isMatch('aa','a*'))
print(isMatch('ab','.*'))
print(isMatch('aab','c*a*b'))




