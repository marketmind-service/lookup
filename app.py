# app.py (inside the lookup ACA)
from fastapi import FastAPI, HTTPException
from state import AgentState
from lookup_agent import lookup_agent

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )