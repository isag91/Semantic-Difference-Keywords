import pandas as pd
import spacy
nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')



#each record in group.json contains the texts belonging to the group concatenated together, as well as a code identifying the group/subcorpus
df=pd.read_json("group.json")


def lemmatizer(text):
    text=text.lower()
    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])


def l_sentences(text):
    doc=nlp(text)
    sentences=[sent.text.strip() for sent in doc.sents]
    return sentences

#each subcorpus is lemmatized and split into sentences
df['lemmatized_text']=df['text'].apply(lemmatizer)
df['lemmatized_text_list']=df['lemmatized_text'].apply(l_sentences)


#########################
#Identify potential SDKs#
#########################

codes=list(df['code'].unique())

#keywords_SE.csv contains the statistical keywords extracted by Sketch Engine when comparing the whole corpus to a general Spanish language corpus
words = 'keywords_SE.csv'
wrd=pd.read_csv(words, skiprows=[0, 1])
keywords=wrd['Item'].to_list()

#filtering keywords who are not nouns and whose keyness score is smaller than one when comparing the full corpus to a reference general language Spanish corpus
wrd=wrd[wrd['Score']>=1.00]
all_words=wrd['Item'].to_list()
kw=[w.text for wl in all_words for w in nlp(wl) if w.pos_ == "NOUN"]


#filtering keywords which are stopwords or do not exist
del_list=['ese', 'ahí', 'nos', 'ni', 'qué', 'l', 're', 'nam', 'c', 'do', 'cia', 'po', 'ó', 'ra', 'to', 'r', 'co', 'no', 'n', 'ca', 'mo', 'ta', 'www', 'nes', 'org', 'acá', 'ah', 'tra', 'so', 'd', 'y', 'pa', 'oea', 'lo', 'le', 'ro', 'ff', 'na', 's', 'pro', 'pue', 'tlc', 'tes', 'mos', 'amo', 'pag', 'pcp', 'ma', 'tos', 'aa', 'blo', 'que', 'tal', 'tar', 't', 'pre', 'im', 'gbi', 'per', 'tro', 'vil', 'lu', 'go', 'el', 'cha', 'ya', 'tas', 'gue', 'nal', 'ne', 'lar', 'así', 'cen', 'zas', 'bre', 'ara', 'ba', 'dis', 'cut', 'em', 'dih', 'or', 'fal', 'dih', 'ria', 'mm', 'vo', 'rra', 'a', 'cos', 'gn', 'se', 'nor', 'pe', 'más', 'za', 'bur', 'uni', 'ses', 'ga', 'gua', 'sdr', 'faz', 'ral', 'cla', 'dor', 'v', 'nas', 'ob', 'tam', 'ras', 'cam', 'up', 'mmh', 'an', 'nue', 'rá', 'í', 'ac', 'ble', 'cer', 'gro', 'rio', 'ri', 'htm', 'sas', 'tre', 'cho', 'i', 'pos', 'fes', 'fun', 'ins', 'm-l', 'bm', 'feu', 'lle', 'gar', 'li', 'nar', 'mu', 'ama', 'dea', 'sm', 'reo', 'zar', 'men', 'tie', 'fac', 'sec', 'in', 'ife', 'pu', 'ón', 'cio', 'moa', 'ini', 'am', 'ol', 'fa', 'afi', 'rue', 'fao', 'ar', 'cc', 'z', 'hh', 'tor', 'eu', 'num', 'cu', 'res', 'ter', 'dd', 'ban', 'bid', 'au', 'we', 'ci', 'of', 'fi', 'sa', 'be', 'jo', 'dió', 'car', 'ría', 'pm', 'der', 'il', 'dic', 'iu', 'ap', 'ei', 'xxi', 'bla', 'ene', 'ia', 'ce', 'via', 'hrs', 'cal', 'e', 'fué', 'can', 'ad', 'fax', 'por', 'ano', 'iv', 'pib', 'sub', 'du', 'g', 'uu', 'sos', 'as', 'j', 'f', 'ee', 'man', 'vas', 'ine', 'ix', 'de', 'w', 'vii', 'fin', 'ay', 'eh', 'eta', 'p', 'ok', 'xi', 'ja', 'm', 'su', 'for', 'aun', 'bus', 'it', 'q', 'lic', 'gil', 'sl', 'on', 've', 'en', 'si', 'té', 'xx', 'yo', 'iva', 'xvi', 'vos', 'us', 'iii', 'gen', 'cd', 'tú', 'mío', 'xix', 'con', 'new', 'ana', 'and', 'o', 'b', 'ex', 'k', 'com', 'h', 'á', 'et', 'ii', 'tv', 'mí', 'él', 'ti', 'muy', 'x', 'asi', 'the', 'mas', 'os', 'me', 'te', 'mi', 'tu', 'file', 're', 'viet']
keywords=[kw for kw in keywords if kw not in del_list]


#count the frequency of each keywords in each subcorpus
codes=list(df['code'].unique())
keyword_count={}
for code in codes:
    print(code)
    nn=0
    keyword_count[code]=defaultdict(int)
    row = df[df['code']==code]
    text=row['lemmatized_text']
    string=str(text.values)
    string=string.replace('- ', '')

    doc=nlp(string)
    list_tokens=[i.text for i in doc]
    
    for kw in keywords:
        nn+=1
        if kw in list_tokens:
            keyword_count[code][kw]=list_tokens.count(kw)

#filtering keywords which appear less than 400 times in the EZLN corpus, and less than 1000 times in the other subcopora combined
kw_count=pd.DataFrame.from_dict(keyword_count).reset_index()
kw_count['other'] = kw_count.drop('EZLN_mex3', axis=1).sum(axis=1)
kw_count_400=kw_count[['index', 'EZLN_mex3', 'other']]
kw_count_400=kw_count_400[kw_count_400['EZLN_mex3']>=400]
kw_count_400=kw_count_400[kw_count_400['other']>=1000]

kw_ref=kw_count_400['index'].to_list()
with open('kw_ref_EZLN_all_400.txt', 'w') as f:
    for line in kw_ref:
        f.write(f"{line}\n")


#########################################################
#Clean corpus and add a reference code to potential SDKs#
#########################################################


df=df[['code', 'lemmatized_text_list']]
df1=df.explode('lemmatized_text_list')


#remove stopwords and "sentences" which are too short to provice context

def cleaning(doc):
    txt = [lemma for lemma in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)

brief_cleaning = (re.sub("[^A-Za-z0-9áéíóúüñÁÉÑÍÓ_']+", ' ', str(row)) for row in df1['lemmatized_text_list'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=300, n_process=-1)]

df_clean = pd.DataFrame({'clean': txt})
df_clean['code']=df1['code'].to_list()
df_clean = df_clean.dropna()

def replace_keywords(vec):
    
    code=vec[0]
    sent=vec[1]
    if code=='EZLN_mex3':
        sent_l=sent.split(' ')
        for kw in kw_ref:
            if kw in sent_l:
                sent_l=list(map(lambda x: x.replace(kw, kw+'_'+code), sent_l))
        new_sent=' '.join(sent_l)
    else:
        new_sent=sent

    return new_sent


df_clean['clean']=df_clean[['code', 'clean']].apply(replace_keywords, axis=1)
df_clean.to_pickle('word2vec_EZLN_others_400.pickle')