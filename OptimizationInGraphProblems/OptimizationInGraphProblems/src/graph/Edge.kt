package graph

class Edge(val startingNode: Node, val endingNode: Node, val cost: Int) {

    init {
        startingNode.outgoingEgdes += this
    }

    override fun toString(): String {
        return "${startingNode.name}${endingNode.name}: $cost"
    }
}