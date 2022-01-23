def maxArea1(height):
    l=0
    r=len(height)-1
    maxarea = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        maxarea = max(maxarea, area)
        if height[l]<=height[r]:
            l+=1
        else:
            r-=1
    print(maxarea)
def maxArea2(height):
    area=[]
    for i in range(len(height)):
        for j in range(i+1,len(height)):
            s=min(height[i],height[j])*(j-i)
            area.append(s)
    print(max(area))
maxArea1([1,4,3,5,6,8])
maxArea2([1,4,3,5,6,8])


