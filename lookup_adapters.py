from state import AgentState, LookupState


def into_lookup_state(parent: AgentState, child: LookupState) -> LookupState:
    return child.model_copy(update={
        "prompt": parent.prompt
    })


def out_of_lookup_state(parent: AgentState, child: LookupState) -> AgentState:
    return parent.model_copy(update={
        "lookup_result": child,
        "route_taken": [*parent.route_taken, "lookup_agent_done"],
    })
