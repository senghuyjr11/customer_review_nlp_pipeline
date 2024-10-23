import uvicorn

if __name__ == "__main__":
    # This will run the FastAPI app at the given path
    uvicorn.run("app.interfaces.api:app", host="127.0.0.1", port=8000, reload=True)
