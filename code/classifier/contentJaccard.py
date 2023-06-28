import jieba
 
def Jaccrad(model, reference):
    terms_reference= jieba.cut(reference)
    terms_model= jieba.cut(model)
    grams_reference = set(terms_reference)
    grams_model = set(terms_model)
    temp=0
    for i in grams_reference:
        if i in grams_model:
            temp=temp+1
    fenmu=len(grams_model)+len(grams_reference)-temp
    jaccard_coefficient=float(temp/fenmu)
    return jaccard_coefficient
 

