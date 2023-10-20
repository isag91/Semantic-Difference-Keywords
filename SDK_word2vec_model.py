import pandas as pd
import multiprocessing
from gensim.models import Word2Vec

######################
#Train word2vec model#
######################

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=20,
                     window=5,
                     sg=1,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


df=pd.read_pickle('word2vec_EZLN_others_400.pickle')
sentences = [row.split() for row in df['clean']]
w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)
w2v_model.save("word2vec_EZLN_all_400.model")

##########################################################
#Sort keywords from smallest to highest cosine similarity#
##########################################################

kw = []
with open('kw_ref_EZLN_all_400.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        kw.append(x)


kw_ezln=[k+'_EZLN_mex3' for k in kw]

for k, k_ezln in zip(kw, kw_ezln):
    try:
        sim=w2v_model.wv.similarity(k, k_ezln)
        l_sim_kw.append(sim)

df=pd.DataFrame(list(zip(kw, kw_ezln, l_sim_kw)), columns=['word', 'word_ezln', 'cosine_similarity'])
df_kw=df.sort_values('cosine_similarity').reset_index(drop=True)
