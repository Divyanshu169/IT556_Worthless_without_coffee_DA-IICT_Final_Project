import csv

res={}
with open('genres1','r') as f:
	for d in f:
		#print(d)
		d=d.split(";")
		for x in d:
			#x=x.replace('\n','')
			if (x.strip() not in res) :
				res.setdefault(x,[])

fin=list(res.keys())
list.sort(fin)
print(fin)
print(len(res))
	
