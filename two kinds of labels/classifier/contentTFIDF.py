from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

document = []

mathFile = "../concept.txt"


mathConcepts = []
for line in open(mathFile, encoding="utf-8"):
    mathConcepts.append(line.strip("\t\n"))
    


mathWikiFile = "../wiki/"


for concept in mathConcepts:
    filename = mathWikiFile + concept + "content.txt"
    f = open(filename, encoding="utf-8")
    doc = f.read()
    sentences = doc.split()
    sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
    doct = [" ".join(sent0) for sent0 in sent_words]
    doct = " ".join(doct)
    document.append(doct)


tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_df=0.6,stop_words=["是", "的"],max_features=10).fit(document)
sparse_result = tfidf_model.transform(document)
print(sparse_result)

ans = sparse_result.todense()
print(ans.shape)