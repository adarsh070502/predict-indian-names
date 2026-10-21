# Preview the Hugging Face Space frontend locally.
# Run:  python preview_frontend.py   -> opens http://127.0.0.1:7860 in your browser
from app import iface

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
