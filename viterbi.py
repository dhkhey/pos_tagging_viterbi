import numpy as np
import re
import nltk

emits = {}  
f = open('emiss_probs.txt')
for line in f:
    parts = line.rstrip().split()
    pos = parts[0]
    word = parts[1]
    eprob = float(parts[2])
    if pos in emits.keys():
        emits[pos][word] = eprob
    else:
        emits[pos] = {word:eprob}
f.close()

t_probs = {}  

f = open('transit_probs.txt')
for line in f:
    parts = line.rstrip().split()
    pos1 = parts[0]
    pos2 = parts[1]
    prob = float(parts[2])
    t_probs[(pos1, pos2)] = prob
f.close()

pos2int = list(emits.keys())

# Most frequent POS baseline
def nltk_baseline(sentence):
    pos_result = "BOS"
    for t in sentence:
        pos_result = pos_result + " " + (nltk.pos_tag([t])[0][1])
    return(pos_result + " EOS")

def viterbi(sentence):
    missing_emitprob =  np.float64(0.0000000001)
    trellis = np.zeros(shape=(len(sentence), len(pos2int)))
    backpointers = np.zeros(shape=(len(sentence), len(pos2int)))
    trellis[0,pos2int.index("BOS")] = 1.0
    
    
    for i in range(1, len(sentence)):
        for j in range(0,len(pos2int)):
            max_path = np.float64(0)
            for k in range(0, len(pos2int)): 
                try:
                    prob = trellis[i-1,k] * t_probs[(pos2int[k], pos2int[j])]
                    if prob > max_path:
                        max_path = prob
                        backpointers[i,j] = k
                except: #TRANSITION PROB DOESNT EXIST
                    pass 
            try:
                trellis[i,j] = max_path * emits[pos2int[j]][sentence[i]]
            except: #EMISSION PROB DOESNT EXIST
                trellis[i,j] = max_path * missing_emitprob
        

    pos_result = "EOS"
    maxprob = trellis[len(sentence)-1,pos2int.index("EOS")]
    maxprobid = backpointers[len(sentence)-1,pos2int.index("EOS")]
    pos = pos2int[int(maxprobid)]
    pos_result = pos + " " + pos_result
   
    for i in range(len(sentence)-2, 0, -1):
        maxprobid = backpointers[i, pos2int.index(pos)]
        pos = pos2int[int(maxprobid)]
        pos_result = pos + " " + pos_result
        
    return pos_result


## Function to find incorrect tags
def correct_tag_count(sq1, sq2):
    correct_count = 0
    numwrong = 0
    if len(sq1) == len(sq2):
        for idx in range(len(sq1)):
            if sq1[idx] == sq2[idx]:
                correct_count += 1
            else:
                numwrong += 1
    return [correct_count, numwrong]
                



#TEST
def run_test():
    # viterbi
    v_numcorrect = 0
    v_numwrong = 0
    # nltk baseline
    n_numcorrect = 0
    n_numwrong = 0

    f = open("test.txt")
    for line in f:

        ## Break each sentence into POS tags and word tokens
        line = re.sub(r'\(', "", line.rstrip())
        line = re.sub(r'\)', "", line.rstrip())
        parts = line.split()

        ## Create strings with words and POS tags
        testsent = "BOS "
        testpos = "BOS "
        for i in range(1, len(parts), 2):
            testpos = testpos + parts[i] + " "
            testsent = testsent + parts[i-1] + " "
        testpos = testpos + "EOS"
        testsent = testsent + "EOS"

        print(testsent)
        print(testpos)


        nltk_result = nltk_baseline(testsent.split()[1:-1])

        sq1 = testpos.split()
        sq2 = nltk_result.split()

        (c, w) = correct_tag_count(sq1, sq2)
        n_numcorrect += c
        n_numwrong += w


        viterbi_result = viterbi(testsent.split())
        # print(viterbi_result + "\n")

        sq1 = testpos.split()
        sq2 = viterbi_result.split()
        (c, w) = correct_tag_count(sq1, sq2)
        v_numcorrect += c
        v_numwrong += w

    f.close()


    ## Print out overall tagging accuracy
    print("Viterbi Accuracy: ", (v_numcorrect / (v_numcorrect+v_numwrong)))
    print("Baseline: ", (n_numcorrect / (n_numcorrect+n_numwrong)))

run_test()



