

import numpy as np
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pyplot as plt


def gen_mask(seka):
    stringas=""
    for i in seka:
        stringas=stringas+"X"
    return stringas

def random_primer(length):
    listas = ['A', 'T', 'G', 'C']
    s=""
    #random hexamer
    seq =np.random.choice(listas, length, p=[0.25, 0.25, 0.25, 0.25])
    return s.join(seq)

def complement(base,probability):
    if base=="A":
        return "T"
    elif base=="T":
        return "A"
    elif base=="C":
        return "G"
    elif base=="G":
        s=""
        return s.join(np.random.choice(["C", "M"], 1, p=[1-probability, probability]))
    elif base=="M":
        return "W"
    elif base=="W":
        return "M"
    else:
        return "ERORR"

def tolist(string):
    listas=[]
    for i in string:
        listas.append(i)
    return listas

def complement_s(sequence):
    stringas=""
    for i in sequence:
        stringas=stringas+complement(i, 0)
    return stringas

def revstring(string):
    st=""
    for i in string:
        st=i+st
    return st



counter=-1

first="TTGATCACGTGGATAGAGGCGCGATTTATTTGGACCCACGGAGGTGCAATGCACCCACCGAGTGAATTTCGCAAGAGATTGCCTCGCGC"

sistema=[]
probability = 0.0001
prime_it =10 # prime every ten iterations
maxiter=10000

class branch():
    def __init__(self, sequence, flankPoz, parent, startOnParent, now, state, akids):
        global counter
        counter=counter+1
        self.seq=sequence
        self.fpoz=flankPoz
        self.pID=parent # sequence is synthesized on
        self.ID=counter # sequence ID created on object creation
        self.start=startOnParent # initial position of primer on parent sequence
        self.end=now
        self.status=state
        self.kids=akids
    def flanking(self):
        if self.fpoz==None:
            return ""
        else:
            return (self.seq)[:(self.fpoz)+1]


    def children(self):
        global sistema
        listofchildren=[]
        number=self.ID

        for i in sistema:
            if i.pID==number and not i.ID==None:
                listofchildren.append([i.ID, i.end, i.start])
        return sorted(listofchildren, key=itemgetter(2), reverse=False)


    def children2(self):
        global sistema
        listofchildren=[]
        number=self.kids

        for i in number:
            listofchildren.append([i, sistema[i].end, sistema[i].start])
        return sorted(listofchildren, key=itemgetter(2), reverse=False)


    def prime(self):
        global sistema
        #pr=["GGACCCA","CACCGAG","AATTTC"]
        #for pradmuo in pr:
        #    seka=self.flanking()
        #    poz=seka.find(pradmuo)

        #sistema.append(branch(revstring(complement_s(pradmuo)), None, self.ID, poz, poz,"active"))
        pradmuo=random_primer(6)
        #seka=self.flanking()
        #priming only free sequence
        seka=self.masked()
        poz=seka.find(pradmuo)
        if poz>0:
            tarpinis=branch(revstring(complement_s(pradmuo)), None, self.ID, poz, poz,"active",[])
            listas=self.kids
            listas.append(tarpinis.ID)
            self.kids=listas
            sistema.append(tarpinis)
        return

    def update(self, iteration):
        global prime_it
        if iteration%prime_it==0:
            self.prime()
        global sistema
        global probability
        if not (self.pID==None) and not (self.status=="terminated"):
                if self.end>0:
                    self.end=self.end-1
                    com=complement((sistema[self.pID].seq)[self.end], probability)
                    self.seq=self.seq+com
                    if com=="M" or com=="W":
                        self.status="terminated"

                else:
                    self.status="terminated"

        #poz=self.children()
        poz=self.children2()
        ## updates position of displaced fragment
        for i in range(1, len(poz)):
            pries=poz[i-1]
            po=poz[i]
            if po[1] in range(pries[1],pries[2]+6):
                sistema[pries[0]].fpoz=pries[2]-po[1]+6

        return

    def masked(self): #masked flanking sequence
        #poz=self.children()
        poz=self.children2()
        seka=self.flanking()
        newstring=seka
        for item in poz:
            if item[1]==0:
                newstring=gen_mask(newstring[:item[2]+6])+newstring[item[2]+6:]
            else:
                newstring=newstring[:item[1]]+gen_mask(newstring[item[1]:item[2]+6])+newstring[item[2]+6:]


        #print "\nSEKA", seka
        #print "NEKA", newstring,"\n"
        return newstring
        #return poz, pradmuo, revstring(complement_s(pradmuo))

sistema.append(branch(first, 70, None, None, None, "terminated",[]))
sistema.append(branch(first, 50, None, None, None, "terminated",[]))
sistema.append(branch(first, 60, None, None, None, "terminated",[]))

sk=-1


for i in range(maxiter):
    probability=probability-probability/(maxiter*1.)
    if i%100==0:
        print(i), probability
    sistema[0].seq=sistema[0].seq+"".join(np.random.choice(["C", "M" ,"G", "A", "T"], 1, p=[0.25*(1-probability),0.25*probability, 0.25, 0.25, 0.25]))
    sistema[0].fpoz=sistema[0].fpoz+1
    sistema[1].seq=sistema[1].seq+"".join(np.random.choice(["C", "M" ,"G", "A", "T"], 1, p=[0.25*(1-probability),0.25*probability, 0.25, 0.25, 0.25]))
    sistema[1].fpoz=sistema[1].fpoz+1
    sistema[2].seq=sistema[2].seq+"".join(np.random.choice(["C", "M" ,"G", "A", "T"], 1, p=[0.25*(1-probability),0.25*probability, 0.25, 0.25, 0.25]))
    sistema[2].fpoz=sistema[2].fpoz+1
    for test in sistema:
        test.update(i)
        #test.prime()

h=[]
for test in sistema:
    sk=sk+1
    #print "##", sk
    #print "SK", test.seq, test.fpoz, test.pID, test.ID, test.start, test.end, test.status
    #print "CO", revstring(complement_s(test.seq))
    #print "FL", test.flanking()
    #print "Ki", test.children(), "Ii", test.children2()
    #print "Mk", test.masked()
    if sk>2:
        h.append(len(test.seq))

#print h

print(h)

plt.hist(h, 50)
plt.show()
