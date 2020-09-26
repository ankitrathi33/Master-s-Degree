package main

import graph.Graph
import graph.Dijkstra
import java.io.File

object Initializer {
    @JvmStatic fun main(args: Array<String>){
        val graph = Graph.load(File("input.graph"))

        println(graph.toString())

//        println(Dijkstra.compute(graph.nodes.first(), graph.nodes.last(), graph))
        val nodePair = Dijkstra.compute(graph.nodes[2], graph.nodes[0], graph)
        println("Cost: ${nodePair.cost}\nPath: ${nodePair.path}")
    }
}