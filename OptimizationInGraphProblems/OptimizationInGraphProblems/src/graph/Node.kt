package graph

class Node(val name: String){

    override fun equals(other: Any?): Boolean {
        if(other is Node)
            return this.name == other.name
        return false
    }

    val outgoingEgdes = ArrayList<Edge>()

    override fun toString(): String {
        return name
    }
}