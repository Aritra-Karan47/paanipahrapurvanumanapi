"""
Application entrypoint for local execution or Gradio launcher.
"""
from src.app.app import demo, bootstrap

if __name__ == "__main__":
    import os
    import uvicorn
    from main import app

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
