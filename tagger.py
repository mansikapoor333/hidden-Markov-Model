import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    model = None
	###################################################
	# Edit here
	###################################################
    initial=[]
    pi=dict()
    for i in train_data:
        '''if i.words[0] not in pi:
            pi[i.words[0]]=1
        else:
            pi[i.words[0]]=+1'''
        initial.append(i.tags[0])
    for w in initial:
        hra=len(initial)
        pi[w]=initial.count(w)/hra
    #print(pi,sum(pi.values()))
    
    
    
    
    em =dict()
    emct= dict()
    trans = dict()
    transct= dict()
    #unique = []
    worddict=dict()
    tagdict=dict()
    wc=0
    tc=0
    for line in train_data:
    #length=len(line)
    #if line:
        words=line.words
        tags=line.tags
        ctr = len(words)
        first = 'START'
        #if first not in unique:
        #    unique.append(first)
        i = 0
        for a in range(len(words)):
            #wrtg=a.rsplit("/",1)
            w = words[a]
            t = tags[a]
            i += 1
            if w not in worddict:
                worddict[w]=wc
                wc+=1
            if t not in tagdict:
                tagdict[t]=tc
                tc+=1
            #if t not in unique:
            #    unique.append(t)
                
            if first in trans:
                if t in trans[first]:
                    trans[first][t] += 1
                    
                else:
                    trans[first][t] = 1
            else:
                trans[first] = dict()
                trans[first][t] = 1
            
            
            if w in em:
                if t in em[w]:
                    em[w][t] += 1
                    
                else:
                    em[w][t] = 1
            else:
                em[w] = dict()
                em[w][t] = 1
                
            if t not in emct:
                emct[t] = 1
                if i != ctr:
                    if t in transct:
                        transct[t] += 1
                        
                    else:
                        transct[t] = 1
            else:
                emct[t] += 1
                if i != ctr:
                    if t in transct:
                        transct[t] += 1
                        
                    else:
                        transct[t] = 1
            first = t
        
                    
    A=np.zeros([len(tagdict),len(tagdict)])
    transitionprob = dict()
    #print(tagdict)
    pi1=np.zeros(len(tagdict))
    for t in trans:
        transitionprob[t] = dict()
        for t2 in trans[t]:
            if t=='START':
                #print(tagdict[t2])
                pi1[tagdict[t2]]=trans[t][t2]/sum(trans[t].values())
            else:
                transitionprob[t][t2] = [trans[t][t2]/sum(trans[t].values()), trans[t][t2]]#[math.log(transition[t][t2]) - math.log(sum(transition[t].values())), transition[t][t2]]
                A[tagdict[t]][tagdict[t2]]=trans[t][t2]/sum(trans[t].values())#transitionprob[t]["transition"] = sum(transition[t].values())

    emissionprob = dict()
    B=np.zeros([len(tagdict),len(worddict)])
    for w in em:
        emissionprob[w] = dict()
        ctr = 0
        for t in em[w]:
            #if t=='START':
            #print(tagdict[t])
            #emissionprob[w][t] = [emission[w][t]/emissioncount[t], emission[w][t]]#[math.log(emission[w][t]) - math.log(emissioncount[t]), emission[w][t]]
            B[tagdict[t]][worddict[w]]=em[w][t]/emct[t]
            #ctr += emission[w][t]
            #emissionprob[w]["total"] = ctr
	 ##################################################
    #for k,v in emission.items():
    #    print (k,v)
    #    break
    #print(worddict)#,emissionprob)
    #print(pi1.shape,A.shape,B.shape,len(tagdict),len(worddict))
    model=HMM(pi1, A, B, worddict, tagdict)
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    
    tagging = []
	###################################################
	# Edit here
	###################################################
    #print(test_data[0].tags,tags)
    #l1=[]
    for text in test_data:
        l1=[]
        for w in text.words:
            l1.append(w)
            if w not in model.obs_dict:
                model.obs_dict[w]=len(model.obs_dict)
                stt=len(model.state_dict)
                l2=[10**(-6)] * stt
                l2=np.array(l2)
                l2=l2.reshape((l2.shape[0],1))
                #print(model.B.shape,l2.shape)
                model.B=np.append(model.B,l2,1)
                #print(model.B.shape)
        path=model.viterbi(l1)
        tagging.append(path)
        #print(path,l1)
    #print(l1)
    #l1=np.array(l1)
    #l1=l1.reshape((1,l1.shape[0]))
    #alpha = model.forward(l1)
    #beta = model.backward(l1)
    #prob1 = model.sequence_prob(l1)
    #prob2 = model.posterior_prob(l1)
    #print(l1.shape)
    #path=model.viterbi(l1)
    #print(path,l1)
    '''print("datsssssss",train_data[0].words,tags)
    initial=[]
    pi=[]
    for i in train_data:
        initial.append(i.words[0])
    for w in initial:
        pi.append(initial.count(w)/len(initial))
    print(pi,sum(pi))'''
    return tagging
