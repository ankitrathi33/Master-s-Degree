package graph

import java.io.File
import java.lang.StringBuilder

class Graph {
    val nodes = ArrayList<Node>()
    val edges = ArrayList<Edge>()

    companion object {
        fun load(f: File): Graph{
            val result = Graph()

            val br = f.bufferedReader()
            val nodes = br.readLine().split(";")
            nodes.forEach {
                result.nodes += Node(it)
            }

            for(i in 0 until nodes.size){
                val weights = br.readLine().split(";")
                for(j in 0 until nodes.size){
                    val cost: Int? = weights[j].trim().toIntOrNull()
                    if(weights[j].trim() != "INF" && weights[j].trim() != "0" && cost != null) {
                        result.edges += Edge(result.nodes[i], result.nodes[j], cost)
                    }
                }
            }

            return result
        }
    }

    override fun toString(): String {
        val sb = StringBuilder()
        sb.append("Nodes:\n")
            .append(nodes.joinToString(separator = ", "))
            .append("\n\nEdges:\n")
            .append(edges.joinToString(separator = ",\n"))

        return sb.toString()
    }
}