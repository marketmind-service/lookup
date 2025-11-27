from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from state import AgentState, LookupState
from lookup_agent import lookup_agent

app = FastAPI(title="Lookup Agent API")


class LookupRequest(BaseModel):
    parent_state: AgentState


class LookupResponse(BaseModel):
    state: AgentState


@app.get("/")
async def root():
    return {"status": "ok", "service": "lookup_agent"}


@app.post("/api/lookup-agent", response_model=LookupResponse)
async def run_lookup(req: LookupRequest):
    try:
        updated_state = await lookup_agent(req.parent_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lookup_agent_error: {e}")

    return LookupResponse(state=updated_state)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
