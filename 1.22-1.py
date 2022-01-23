def twoSum(nums,target):
    length=len(nums)
    for i in range(length):
        for j in range(i+1,length):
            if nums[i]+nums[j]==target:
                print([i,j])
twoSum([2,7,11,15],9)
twoSum([1,2,3,4],5)
