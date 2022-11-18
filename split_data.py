import sys
with open(sys.argv[1], "r") as f1:
	l1 = f1.readlines()

with open(sys.argv[2], "r") as f2:
	l2 = f2.readlines()

w_dict={}
it = 0
w_dict[0] = []
for line1, line2 in zip(l1,l2):
	line1=line1.strip()
	if line1:
		s1, s2 = line1.split()
		s1, s2 = s1.strip(), s2.strip()
		s3 = line2.strip()
		w_dict[it].append((s1,s2,s3))
	else:
		it += 1
		w_dict[it] = []
	
# import pdb
# pdb.set_trace()
import random
sample_it = list(range(it-1))
random.shuffle(sample_it)

sample_rat = float(sys.argv[3])
T = int(sample_rat * it)

f1 = open(sys.argv[1]+"_"+str(sys.argv[3])+".txt", "w")
f2 = open(sys.argv[2]+"_"+str(sys.argv[3])+".txt", "w")
for i in range(T):
	l_tmp = w_dict[sample_it[i]]
	for it in l_tmp:
		s1,s2,s3 = it
		f1.write(s1+"\t"+s2+"\n")
		f2.write(str(i)+"\n")
	if i != T-1:
		f1.write("\n")
		f2.write("\n")
f1.close()
f2.close()
