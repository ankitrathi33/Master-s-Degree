package graph

data class NodePair(val node: Node, var cost: Int = Int.MAX_VALUE / 2, var visited: Boolean = false, var path: String = "")