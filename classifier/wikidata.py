import wikipedia
concepts = []
for line in open("concept.txt","r", encoding="utf-8"):
    concepts.append(line.strip("\t\n"))




for concept in concepts:
    print(concept)
    ny = wikipedia.page(concept)
    if ny == None:
        continue
    filename = "..wiki/" + concept
    
    f = open(filename+"_summary.txt","w", encoding="utf-8")
    f.write(ny.summary)
    f.close()

    f = open(filename + "_title.txt","w", encoding="utf-8")
    f.write(ny.title)
    f.close()

    f = open(filename + "_content.txt","w", encoding="utf-8")
    f.write(ny.content)
    f.close()

    f = open(filename + "_links.txt","w", encoding="utf-8")
    f.write("\n".join(ny.links))
    f.close()

    f = open(filename + "_categories.txt","w", encoding="utf-8")
    f.write("\n".join(ny.categories))
    f.close()

    

