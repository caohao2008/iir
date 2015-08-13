#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Labeled Latent Dirichlet Allocation
# This code is available under the MIT License.
# (c)2010 Nakatani Shuyo / Cybozu Labs Inc.
# refer to Ramage+, Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora(EMNLP2009)

from optparse import OptionParser
import sys, re, numpy, json
import thread
import threading

reload(sys)  # Reload does the trick!  
sys.setdefaultencoding('UTF8')

def load_corpus(filename):
    corpus = []
    labels = []
    labelmap = dict()
    f = open(filename, 'r')
    for line in f:
        mt = re.match(r'\[(.+?)\](.+)', line)
        if mt:
            label = mt.group(1).split(',')
            #print "label = ",label
            for x in label: labelmap[x] = 1
            line = mt.group(2)
        else:
            label = None
        #print "label = ",label
        #doc = re.findall(r'\w+(?:\'\w+)?',line.lower())
        #doc = re.findall(r'[\w|\u4e00-\u9fff]+(?:\'[\w|\u4e00-\u9fff]+)?',line.lower())
        docs = line.split(" ")
        #print "doc = ",docs
	doc = []
	for d in docs:
		if d!="":
			doc.append(d)
			#print json.dumps(d).decode("unicode-escape")
        #print "doc = ",doc
	if len(doc)>0:
            corpus.append(doc)
            labels.append(label)
    f.close()
    return labelmap.keys(), corpus, labels

class inference_thread(threading.Thread):
    def __init__(self, llda, m, doc, label, V):
        threading.Thread.__init__(self)
        self.doc = doc
        self.label = label
        self.llda = llda
        self.V = V
        self.m = m
   
    def run(self):
        self.inference_doc(self.llda,self.m,self.doc,self.label,self.V)
         
    def inference_doc(self,llda,m,doc,label,V):
        for n in range(len(doc)):
            t = doc[n]
            z = llda.z_m_n[m][n]
            llda.n_m_z[m, z] -= 1
            llda.n_z_t[z, t] -= 1
            llda.n_z[z] -= 1
    
            denom_a = llda.n_m_z[m].sum() + llda.K * llda.alpha
            denom_b = llda.n_z_t.sum(axis=1) + V * llda.beta
            p_z = label * (llda.n_z_t[:, t] + llda.beta) / denom_b * (llda.n_m_z[m] + llda.alpha) / denom_a
            new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
    
            llda.z_m_n[m][n] = new_z
            llda.n_m_z[m, new_z] += 1
            llda.n_z_t[new_z, t] += 1
            llda.n_z[new_z] += 1


class LLDA:
    def __init__(self, K, alpha, beta):
        #self.K = K
        self.alpha = alpha
        self.beta = beta

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label):
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        labelset.insert(0, "common")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.labels = numpy.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)
	print "corpus size :",M
        V = len(self.vocas)
	print "word size :",V
	print "主题数 :",self.K
	
	#文档数*每个文档词个数的矩阵
        self.z_m_n = []
        #n_m_z为文档数*主题数的矩阵
	self.n_m_z = numpy.zeros((M, self.K), dtype=int)
        #n_z_t为主题数*词数的矩阵
        self.n_z_t = numpy.zeros((self.K, V), dtype=int)
        #n_z为主题数的数组
        self.n_z = numpy.zeros(self.K, dtype=int)

        #初始化，此处复杂度为文档数*词数*主题数
	for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            #z_n = [label[x] for x in numpy.random.randint(len(label), size=N_m)]
            #初始化每个文档的词向量数组
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            #print z_n
            self.z_m_n.append(z_n)
	    #初始化矩阵每维为1
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

   
    def inference(self):
        V = len(self.vocas)
        #complexity : 文档数 * doc数 * label数* 词总数
        threads = []
	for m,doc,label in zip(range(len(self.docs)), self.docs, self.labels):
            #threads = []
	    thread = inference_thread(self,m,doc,label,V)
            thread.start()
            threads.append(thread) 
        for t in threads:
            t.join()
                
            
                

    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
            #print "doc=",doc,"theta=",theta
        return numpy.exp(log_per / N)

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.001)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    (options, args) = parser.parse_args()
    if not options.filename: parser.error("need corpus filename(-f)")

    labelset, corpus, labels = load_corpus(options.filename)

    llda = LLDA(options.K, options.alpha, options.beta)
    llda.set_corpus(labelset, corpus, labels)

    for i in range(options.iteration):
        sys.stderr.write("-- %d " % (i + 1))
        llda.inference()
	##compute perplexity every round
	#print "perplexity=",llda.perplexity()
    print ""
    #print llda.z_m_n

    phi = llda.phi()
    for v, voca in enumerate(llda.vocas):
        print ','.join([voca]+[str(x) for x in llda.n_z_t[:,v]])
        print ','.join([voca]+[str(x) for x in phi[:,v]])
    for k, label in enumerate(labelset):
        print "\n-- label %d : %s" % (k, label)
        for w in numpy.argsort(-phi[k])[:20]:
            print "%s: %.4f" % (llda.vocas[w], phi[k,w])
    #print phi
    #print llda.theta()
if __name__ == "__main__":
    main()
