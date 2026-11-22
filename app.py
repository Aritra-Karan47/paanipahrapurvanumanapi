"""
Hugging Face Space entrypoint.
Simply launches the Gradio app defined under src/app/app.py.
"""
from src.app.app import demo

if __name__ == "__main__":
    demo.launch()