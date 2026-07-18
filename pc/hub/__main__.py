import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("hub.app:app",
                host=os.environ.get("RECALL_HOST", "127.0.0.1"),
                port=int(os.environ.get("RECALL_PORT", "8000")))
