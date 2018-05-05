import csv

res=[]
cnt=0
with open('ratings.csv','r') as in1,open('FinalData.csv','r')as in2,open('FinalRatings.csv','w')as out:
	for x in in2:
		x=x.split(',')
		res.append(x[0])
	for x in in1:
		x=x.split(',')
		if x[1] in res:
			r= x[0]+','+x[1]+','+x[2]
			out.write(r)
			cnt+=1
print(cnt)
print(len(res))
'''
with open('FinalRatings.csv','r')as in1, open('FinalRatings1.csv','w')as out:
	for x in in1:
		x=x.split(',')
		#print(x[1])
		if(int(x[1])<1000):
			print(x[0]+x[1]+x[2])
			out.write(x[0]+x[1]+x[2])
'''
