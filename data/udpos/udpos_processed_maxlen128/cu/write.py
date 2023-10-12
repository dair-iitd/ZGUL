import sys
f = open(sys.argv[1])
st = f.read()
f.close()
lines = st.split("\n\n")

with open(sys.argv[2], "w") as f:
	for line in lines:
		words_lab = line.strip().split("\n")
		words = []
		for item in words_lab:
			words.append(item.split("\t")[0])
		line = " ".join(words)
		f.write(line+"\n")