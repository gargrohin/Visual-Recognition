images =[]
f = open("images.txt",'r')
for x in f:
    images.append(x)
    #print(x)

for i in range(1212):
    images[i] = images[i].rsplit('\n')[0]


out =[]
f = open("fine_results.txt",'r')
for x in f:
    out.append(x)
    #print(x)

for i in range(1213):
    out[i] = out[i].rsplit('\n')[0]

strout=[]
for i in range(1213):
    stro = images[i] +" " + out[i].rsplit('@')[0] + " " + out[i] + "\n"
    strout.append(stro)

fout = open("output.txt","w")
for i in range(1213):
    fout.write(strout[i])

