"""AnvilAgent CLI entry point."""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("anvil")


def main():
    parser = argparse.ArgumentParser(
        description="AnvilAgent - Local-first multi-agent orchestrator for AMD APU"
    )
    parser.add_argument("--api", action="store_true", help="Start the REST API server")
    parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="API server port (default: 8080)")
    parser.add_argument("--orchestrate", action="store_true", help="Run demo orchestration (mock)")
    parser.add_argument("--llm", action="store_true", help="Run orchestration with real LLM")
    parser.add_argument("--pipeline", action="store_true", help="Run pipeline (LLM + KV + sandbox)")
    parser.add_argument("--chain", action="store_true", help="Run multi-agent chain")
    parser.add_argument("--inference-host", default="127.0.0.1", help="llama-server host")
    parser.add_argument("--inference-port", type=int, default=8081, help="llama-server port")
    parser.add_argument("--detect", action="store_true", help="Detect and profile hardware")
    parser.add_argument("--setup", action="store_true", help="Print setup instructions")
    parser.add_argument("--version", action="store_true", help="Print version")

    args = parser.parse_args()

    from anvil.cli.commands import (
        cmd_api,
        cmd_chain,
        cmd_detect,
        cmd_llm,
        cmd_orchestrate,
        cmd_pipeline,
        cmd_setup,
        cmd_version,
    )

    if args.version:
        cmd_version()
    elif args.setup:
        cmd_setup()
    elif args.detect:
        cmd_detect()
    elif args.api:
        cmd_api(args.host, args.port)
    elif args.orchestrate:
        cmd_orchestrate()
    elif args.llm:
        cmd_llm(args.inference_host, args.inference_port)
    elif args.pipeline:
        cmd_pipeline(args.inference_host, args.inference_port)
    elif args.chain:
        cmd_chain(args.inference_host, args.inference_port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
