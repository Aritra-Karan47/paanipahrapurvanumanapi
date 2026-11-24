import os

# Disable Gradio telemetry and mounting during automated test discovery and execution
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MOUNT_GRADIO"] = "False"
