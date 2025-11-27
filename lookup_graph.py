from langgraph.graph import StateGraph, END
from state import LookupState
from parse_input import parse_input
from search import search


def create_lookup_graph():
    graph = StateGraph(LookupState)
    graph.set_entry_point("parse_input")

    graph.add_node("parse_input", parse_input)
    graph.add_node("search", search)

    graph.add_edge("parse_input", "search")
    graph.add_edge("search", END)

    return graph.compile()
