import os
path='crawled'
ls=os.listdir(path)

setName={}

for x in ls:
  if x[:-6] not in setName:
    setName[x[:-6]]=[]
  setName[x[:-6]].append(x[-6:-4])

for x in setName:
  for (i,y) in enumerate(setName[x]):
    setName[x][i]='a'+y if int(y)>20 else 'b'+y
  setName[x].sort()
  setName[x]=[y[1:] for y in setName[x]]

for x in setName:
  with open(x+'.txt','w',encoding='utf8') as f:
    for y in setName[x]:
      with open(path+'/'+x+y+'.txt','r',encoding='utf8') as g:
        text=g.read()
      f.write('\n'+text)
      f.flush()
