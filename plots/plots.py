import matplotlib.pyplot as plt
import numpy as np

output=open("log.txt","r")
lines=output.readlines()
sgdacc=list()
nbacc=list()
pacacc=list()

sgdf1=list()
nbf1=list()
pacf1=list()

sgdp=list()
nbp=list()
pacp=list()

sgdaccwo=list()
nbaccwo=list()
pacaccwo=list()

for l in lines:
	l=l.strip()
	vals=list(l.split(" "))
	
	#index 0, 1, 2 are accuracies 
	#index 3, 4, 5 are f1 scores 
	#index 6, 7,8 are predicision scores 
	sgdacc.append(float(vals[0]))
	nbacc.append(float(vals[1]))
	pacacc.append(float(vals[2]))
	
	sgdf1.append(float(vals[3]))
	nbf1.append(float(vals[4]))
	pacf1.append(float(vals[5]))
	
	sgdp.append(float(vals[6]))
	nbp.append(float(vals[7]))
	pacp.append(float(vals[8]))
	
	#like the previous line, do for all other indices till 8 

wostopw=open("log100WO.txt","r")
wolines=wostopw.readlines()
for l in wolines:
	l=l.strip()
	vals=list(l.split(" "))
	sgdaccwo.append(float(vals[0]))
	nbaccwo.append(float(vals[1]))
	pacaccwo.append(float(vals[2]))

sgdacc10k=list()
nbacc10k=list()
pacacc10k=list()

batch10k=open("log10k.txt","r")
lines10k=batch10k.readlines()
for l in lines10k:
	l=l.strip()
	vals=list(l.split())

	sgdacc10k.append(float(vals[0]))
	#sgdp10k.append(float(vals[6]))
	
	nbacc10k.append(float(vals[1]))
	#nbp10k.append(float(vals[7]))
	
	pacacc10k.append(float(vals[2]))
	#pacp10k.append(float(vals[8]))
batch10k.close()

batch8k=open("log8k.txt","r")	
sgdacc8k=list()
nbacc8k=list()
pacacc8k=list()
lines8k=batch8k.readlines()
for l in lines8k:
	l=l.strip()
	vals=list(l.split())

	sgdacc8k.append(float(vals[0]))
	#sgdp10k.append(float(vals[6]))
	
	nbacc8k.append(float(vals[1]))
	#nbp10k.append(float(vals[7]))
	
	pacacc8k.append(float(vals[2]))
	#pacp10k.append(float(vals[8]))
batch8k.close()

batch6k=open("log6k.txt","r")	
sgdacc6k=list()
nbacc6k=list()
pacacc6k=list()
lines6k=batch6k.readlines()
for l in lines6k:
	l=l.strip()
	vals=list(l.split())

	sgdacc6k.append(float(vals[0]))
	#sgdp10k.append(float(vals[6]))
	
	nbacc6k.append(float(vals[1]))
	#nbp10k.append(float(vals[7]))
	
	pacacc6k.append(float(vals[2]))
	#pacp10k.append(float(vals[8]))
batch6k.close()

batches=[100,6000,8000,10000]

avgaccsgd=[np.average(np.array(sgdacc)), np.average(np.array(sgdacc6k)), np.average(np.array(sgdacc8k)),np.average(np.array(sgdacc10k))]
avgaccnb=[np.average(np.array(nbacc)), np.average(np.array(nbacc6k)),np.average(np.array(nbacc8k)),np.average(np.array(nbacc10k))]
avgaccpac=[np.average(np.array(pacacc)),np.average(np.array(pacacc6k)), np.average(np.array(pacacc8k)),np.average(np.array(pacacc10k))]

plt.scatter(np.array(batches),np.array(avgaccsgd))
plt.title("Accuracies for different batch size:SGDC")
plt.ylabel("Average Accuracy")
plt.xlabel("Batch Size")
plt.show()
plt.savefig('accvbatch1.png')

plt.scatter(np.array(batches),np.array(avgaccnb))
plt.title("Accuracies for different batch size:NB")
plt.ylabel("Average Accuracy")
plt.xlabel("Batch Size")
plt.show()
plt.savefig('accvbatch2.png')

plt.scatter(np.array(batches),np.array(avgaccpac))
plt.title("Accuracies for different batch size:PAC")
plt.ylabel("Average Accuracy")
plt.xlabel("Batch Size")
plt.show()
plt.savefig('accvbatch3.png')


#precision and F1 score relationship for SGD classifier
plt.scatter(np.array(sgdf1),np.array(sgdp))
plt.title("Accuracies and F1 Score:SGDC")
plt.ylabel("Precision")
plt.xlabel("F1 Score")
plt.show()
plt.savefig('accvf11.png')

plt.scatter(np.array(nbf1),np.array(nbp))
plt.title("Accuracies and F1 Score:NB")
plt.ylabel("Precision")
plt.xlabel("F1 Score")
plt.show()
plt.savefig('accvf12.png')

plt.scatter(np.array(pacf1),np.array(pacp))
plt.title("Accuracies and F1 Score:PAC")
plt.ylabel("Precision")
plt.xlabel("F1 Score")
plt.show()
plt.savefig('accvf13.png')

plt.plot(np.array(pacacc),linestyle='solid')
plt.plot(np.array(pacaccwo),linestyle='dashed')
plt.title("Accuracies with (solid) and without stopwords (dashed): PAC")
plt.xlabel("Index of datapoints")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('accvsw1.png')

plt.plot(np.array(nbacc),linestyle='solid')
plt.plot(np.array(nbaccwo),linestyle='dashed')
plt.title("Accuracies with (solid) and without stopwords (dashed):NB")
plt.xlabel("Index of datapoints")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('accvsw2.png')

plt.plot(np.array(sgdacc),linestyle='solid')
plt.plot(np.array(sgdaccwo),linestyle='dashed')
plt.title("Accuracies with (solid) and without stopwords (dashed): SGDC")
plt.xlabel("Index of datapoints")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('accvsw3.png')


wostopw.close()
output.close()

