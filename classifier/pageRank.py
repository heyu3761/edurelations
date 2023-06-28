from pygraph.classes.digraph import digraph
import numpy as np
import jieba


class PRIterator:
    __doc__ =

    def __init__(self, dg):
        self.damping_factor = 0.85
        self.max_iterations = 100
        self.min_delta = 0.00001
        self.graph = dg

    def page_rank(self):
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)
        damping_value = (1.0 - self.damping_factor) / graph_size

        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):
                    rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                rank += damping_value
                change += abs(page_rank[node] - rank)
                page_rank[node] = rank

            print("This is NO.%s iteration" % (i + 1))
            print(page_rank)

            if change < self.min_delta:
                flag = True
                break
        if flag:
            print("finished in %s iterations!" % node)
        else:
            print("finished out of 100 iterations!")
        return page_rank


if __name__ == '__main__':
    dg = digraph()
    # concept
    index = 0
    concepts = []
    conceptDic = {}
    document = []
    WikiFile = "../../canusedata/matrix/wiki/"
    for line in open("../../canusedata/matrix/concept.txt", encoding="utf-8"):
        concept = line.strip("\t\n")
        concepts.append(concept)
        conceptDic[concept] = index
        index = index + 1
    
    
    con = set(concepts)
    dg.add_nodes(con)
    
    
    for concept in concepts:
        if concept == "CLIENT/SERVER MODEL":
            concept = "client_server model"
        if concept == "TRANSMISSION CONTROL PROTOCOL/INTERNET PROTOCOL NETWORK":
            concept = "TRANSMISSION CONTROL PROTOCOL_INTERNET PROTOCOL NETWORK"  
        filename = WikiFile + concept + "_content.txt"
        f = open(filename, encoding="utf-8")
        doc = f.read()
        document.append(doc)
        

    cnt = len(document)
    edges = []
    for i in range(cnt):
        for j in range(cnt):
            tmp = document[j].count(concepts[i])
            if tmp>0:
                edges.append((concepts[i],concepts[j]))

    addEgde = set(edges)
    for ae in addEgde:
        dg.add_edge(ae)
    pr = PRIterator(dg)
    page_ranks = pr.page_rank()
    pair = np.zeros((cnt*cnt,1))
    for i in range(cnt):
        for j in range(cnt):
            pair[j * cnt + i][0] = format(page_ranks.get(concepts[i]) - page_ranks.get(concepts[j]),".15f")
    np.savetxt("pair.npy",pair)
