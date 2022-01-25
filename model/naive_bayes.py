
from cProfile import label
from email.policy import default
from typing import List,Dict
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


# https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf

class BernoulliNB(object):
    """
    Bernoulli document model 
    A document is represented by a feature vector with binary elements 
    taking value 1 if the corresponding word is present in the document 
    and 0 if the word is not present 
    """    
    def __init__(self,X:List[List[int]],y:List[int],smoothing=1) -> None:
        assert len(X) == len(y)
        self._feature_dim=X.shape[1] # vocab size 
        self._X=X
        self._y=y
        self._smoothing=smoothing
        self._N,self._N_k,self._n_k_w_t,self._d_k=self._extract_information(X,y)
        self._l_k=defaultdict(list) # The likelihood P(w_t|C_k)
        self._p_k=defaultdict(float) # The priors P(C_k)

    def _extract_information(self,X,y):
        N=len(X)  # The total number of documents
        N_k=defaultdict(int) # The number of documents labelled with class C=k, for k=1,...,K
        n_k_w_t=defaultdict(lambda : defaultdict(int)) #The number of documents of class C=k containing word w_t for every class and for each word in vocab
        d_k=defaultdict(list) # The vector of document class C=k 
        for doc_embedding,label in zip(X,y):
            N_k[label]+=1
            d_k[label].append(doc_embedding)
        for label,docs_embedding in d_k.items():
            n_k_w_t[label]=np.array(docs_embedding).sum(axis=0)
        return N,N_k,n_k_w_t,d_k
               
    def fit(self):
        """
        Estimate the likelihoods, priors
        P(w_t|C_k)=n_k(w_t)/N_k : likelihood
        P(C_k)=N_k/N : prior
        """
        for label in self._y:
            self._l_k[label]=(self._n_k_w_t[label]+self._smoothing)/(self._N_k[label]+self._smoothing*self._feature_dim)
            self._p_k[label]=self._N_k[label]/self._N
           
        
    def predict(self,X:List[List[int]]):
        preds=[]
        probs=[0]*len(np.unique(self._y))
        for doc in tqdm(X):
            for label in sorted(np.unique(self._y)):
                probs[label]=self._p_k[label]*np.prod(self._l_k[label]*doc + (1-doc)*(1-self._l_k[label]))
            preds.append(np.argmax(probs))
        
        return np.array(preds)


def gen_bernoulli_embedding(X:List[str],vocab:Dict[str,int]):
    """
    Generate feature vectors with binary elements 
    taking value 1 if the corresponding word is present in the document 
    and 0 if the word is not present 
    """
    embeddings=[]
    for doc in X:
        doc_embedding=[0]*len(vocab)
        for word in doc.split():
            if word in vocab.keys():
                doc_embedding[vocab[word]]=1
        embeddings.append(doc_embedding)
    return np.array(embeddings)

def main():
        # Dataset 
    train_df=pd.read_csv('data/IMDB/train.csv')
    test_df=pd.read_csv('data/IMDB/test.csv')

    x_train,y_train=train_df['review'],train_df['sentiment']
    x_test,y_test=test_df['review'],test_df['sentiment']

    
    cv=CountVectorizer(lowercase=True,analyzer="word",stop_words="english",min_df=50).fit(x_train.tolist())
    if model_tp =='Bernoulli':
        x_train_embedding=gen_bernoulli_embedding(x_train,cv.vocabulary_)
        x_test_embedding=gen_bernoulli_embedding(x_test,cv.vocabulary_)
        # Train
        model=BernoulliNB(X=x_train_embedding,y=y_train)
        model.fit()
        # Test
        y_preds=model.predict(x_test_embedding)
        print(f'Accuracy : {(y_preds==y_test).sum()/len(y_preds)}')

    elif model_tp =="Multinomial":
        pass


if __name__ == "__main__":
    global model_tp 
    
    model_tp='Bernoulli'
    main()


