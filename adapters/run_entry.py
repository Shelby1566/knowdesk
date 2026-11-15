import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[2] / "tau2-bench/.env", override=True)

from adapters.tau2_adapter import register_knowdesk_components
from tau2.cli import main as tau2_cli_main

if __name__ == "__main__":
    register_knowdesk_components()
    sys.exit(tau2_cli_main())
