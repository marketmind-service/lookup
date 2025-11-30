from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from state import AgentState, LookupState
from lookup_agent import lookup_agent
from search import search

app = FastAPI(title="Lookup Agent API")


@app.get("/")
async def root():
    return {"status": "ok", "service": "lookup_agent"}


@app.post("/api/lookup-agent", response_model=AgentState)
async def run_lookup(state: AgentState):
    try:
        updated_state = await lookup_agent(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lookup_agent_error: {e}")

    return updated_state


class DirectLookupRequest(BaseModel):
    company: str
    period: str
    interval: str


@app.post("/api/lookup", response_model=LookupState)
async def direct_lookup(req: DirectLookupRequest):
    in_state = LookupState(
        company=req.company,
        period=req.period,
        interval=req.interval,
    )

    try:
        out_state = search(in_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lookup_error: {e}")

    if out_state.error:
        raise HTTPException(status_code=400, detail=out_state.error)

    return out_state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
