from typing import cast
from langchain_core.runnables import RunnableConfig
from state import AgentState, LookupState
from lookup_graph import create_lookup_graph
from lookup_adapters import into_lookup_state, out_of_lookup_state


async def lookup_agent(parent: AgentState):
    in_state = into_lookup_state(parent, LookupState())
    raw = await create_lookup_graph().ainvoke(
        in_state,
        config=cast(RunnableConfig, cast(object, {"recursion_limit": 100}))
    )

    out_state = out_of_lookup_state(parent, LookupState(**raw))
    return out_state
