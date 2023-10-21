# Semantic-Difference-Keywords

This is the repository accompanying the paper "From Diachronic to Contextual Lexical Semantic Change: Introducing Semantic Difference Keywords (SDKs) for Discourse Studies". 4th International Workshop on Computational Approaches to Historical Language Change 2023 (LChange'23), EMNLP 2023.

The corpus used for the case study was assembled from three sources:

1. **The CeDeMa archive (Centro de Documentacion de los movimientos armados)**\
URL: https://cedema.org/digital_items \
All documents issued by a movement, written in Spanish, and dated from 1953 onward were selected. Documents which were available in formats other than plain text where converted to plain text. PDF files which needed to be OCRed were OCRed by using Tesseract and Google Vision.[^2]

2. **The archive of the 26th of July Movement (the leading organisation of the Cuban Revolution) and the Castro regime**\
URL: http://www.fidelcastro.cu/es/biblioteca/documentos/coleccion/todas

3. **The archive of the EZLN (Zapatista Army of National Liberation**\
URL: https://enlacezapatista.ezln.org.mx/category/comunicado/ \
All the original Spanish texts were selected. 

Although this data is publicly available, we do not have permission to redistribute it. However, the embeddings of the Word2Vec model are available here: https://lindat.mff.cuni.cz/ *Exact link TBD (dataset submission currently under review)*.

Here is a short overview of the files in this directory.

| Filename    | Content |
| -------- | ------- |
| keywords_SE.csv | This file was obtained by comparing the entire corpus with the general Spanish language corpus esTenTen on Sketch Engine. The keyness scores are computed with the simple maths keyness metric.[^1]   |
| SDK_preprocessing.py  | The script for the preprocessing steps of the raw text data and the statistical keyness data obtained from Sketch Engine. The text data is lemmatized and split into sentences. Statistical keywords which are not nouns, are stop words or do not exist are discarded. In addition, they are filtered according to a keyness threshold, and, after a frequency count, frequency values in the target and reference corpora. Then, the contextual reference is appended to words selected as potential SDKs.|
| kw_ref_EZLN_all_400.txt | The list of potential SDKs obtained from SDK_preprocessing.py|
| pd_kw_count.json | Frequency values of the potential SDKs in the target and reference copora obtained from SDK_preprocessing.py |
| SDK_word2vec.py | The script for training the Word2Vec models (including hyperparameters) and outputting the SDKs from smallest to highest cosine similarity.|

[^1]: Adam Kilgarriff. 2009. Simple maths for keywords. In Proc. Corpus Linguistics, volume 6.
[^2]: Isabelle Gribomont, "OCR with Google Vision API and Tesseract," Programming Historian 12 (2023), https://doi.org/10.46430/phen0109.
