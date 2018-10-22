def pageRank(graph,iterations=50,d = 0.15):
    graphRank = initGraph(graph)
    for i in range(0,iterations):
        newPR = {}
        for node in graph:
            newPR[node] = d/len(graph) + (1-d)*(prestigeCalc(graphRank,graph[node],graph))
        graphRank = newPR
    return graphRank

#def pageRankImproved(graph,iterations=4,d = 0.15):
#    graphRank = initGraph(graph)
#    for i in range(0,iterations):
#        newPR = {}
#        for node in graph:
#            #new formula
#        graphRank = newPR
#    return graphRank

def prestigeCalc(graphRank,u,graph):
    result=0
    for node in u:
        result+=graphRank[node]/len(graph[node])
    return result

def initGraph(graph):
    graphRank = {}
    for node in graph:
        graphRank[node] = 1/len(graph)
    return graphRank

def prior(graph, node):
    return null
