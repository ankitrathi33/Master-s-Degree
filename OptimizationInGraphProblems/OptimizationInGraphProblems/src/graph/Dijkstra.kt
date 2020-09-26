package graph

object Dijkstra {

    val costTable = ArrayList<NodePair>()

    fun compute(startingNode: Node, targetNode: Node ,graph: Graph): NodePair{
        costTable.clear()
        graph.nodes.forEach {
            costTable += NodePair(it)
        }

        computeForStartingNode(startingNode)

        var nodePair = getNextNodePair()
        while(nodePair != null){
            computeForNode(nodePair)
            nodePair = getNextNodePair()
        }

        return getNodePair(targetNode)!!
    }

    private fun getNextNodePair(): NodePair?{
        var result: NodePair? = null
        for(it in costTable){
            if(!it.visited && (result == null || result.cost > it.cost)){
                result = it
            }
        }
        return result
    }

    private fun computeForStartingNode(node: Node){
        val nodePair = getNodePair(node)
        nodePair!!.cost = 0
        nodePair.path = node.name
        computeForNode(nodePair)
    }

    private fun computeForNode(nodePair: NodePair){
        nodePair.node.outgoingEgdes.forEach {edge ->
            val endingPair = getNodePair(edge.endingNode)
            if(endingPair!!.cost > nodePair.cost + edge.cost) {
                endingPair.cost = nodePair.cost + edge.cost
                endingPair.path = "${nodePair.path}${endingPair.node.name}"
            }
        }
        nodePair.visited = true
    }

    private fun getNodePair(node: Node): NodePair?{
        costTable.forEach { nodePair ->
            if(nodePair.node == node)
                return nodePair
        }
        return null
    }
}