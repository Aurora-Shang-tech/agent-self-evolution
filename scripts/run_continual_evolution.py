
"""Command-line entry for frozen continual ALFWorld evolution runs."""

import argparse

from src.evolution.controller import EvolutionController


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--stage",
        choices=("plan", "run", "status", "rollback"),
        default="plan",
    )

    parser.add_argument(
        "--run-name",
        default="dynamic_v1",
    )

    parser.add_argument(
        "--mode",
        choices=("fixed", "append", "dynamic"),
        default="dynamic",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="1 round, 2 learning tasks, 2 validation tasks",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--version",
        help="Previously accepted version for rollback",
    )

    args = parser.parse_args()

    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("--max-rounds must be positive")

    controller = EvolutionController(
        run_name=args.run_name,
        mode=args.mode,
        pilot=args.pilot,
    )

    if args.stage == "plan":
        state = controller.initialize()

        print(f"Frozen run: {controller.home}")
        print(f"Mode: {controller.mode}; pilot: {controller.pilot}")
        print(f"Learning tasks per round: {controller.learn_limit}")
        print(f"Validation tasks: {controller.validation_limit}")
        print(f"Current version: {state['active_version']}")
        print("No model calls were made.")

    elif args.stage == "run":
        controller.run(args.max_rounds)

    elif args.stage == "status":
        controller.status()

    else:
        if not args.version:
            parser.error(
                "--version is required for rollback"
            )

        controller.rollback(args.version)


if __name__ == "__main__":
    main()

