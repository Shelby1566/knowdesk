import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[3] / "tau2-bench/.env", override=True)

from knowdesk.adapters.tau2_adapter import register_knowdesk_components
from tau2.cli import main as tau2_cli_main

if __name__ == "__main__":
    register_knowdesk_components()
    sys.exit(tau2_cli_main())
