"""Simple Assistant Workflow CLI Entry Point."""

import argparse

from langchain_core.messages import HumanMessage

from workflow.graph import app, run_simple_assistant


def print_graph():
    """Print the workflow graph in mermaid format."""
    print("\n" + "=" * 60)
    print("Workflow Graph:")
    print("=" * 60)
    graph = app.get_graph()
    print(graph.draw_mermaid())
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Simple Assistant Workflow")
    parser.add_argument("--show-graph", action="store_true", help="Show workflow graph")
    parser.add_argument("--user-input", type=str, help="User input for the workflow")
    parser.add_argument("--code", type=str, default="", help="Code snippet (optional)")
    parser.add_argument("--error", type=str, default="", help="Error message (optional)")
    parser.add_argument("--expected", type=str, default="", help="Expected behavior (optional)")
    parser.add_argument("--language", type=str, default="Python", help="Programming language")

    args = parser.parse_args()

    if args.show_graph:
        print_graph()
        return

    messages = []

    if args.user_input:
        messages.append(HumanMessage(content=args.user_input))
        result = run_simple_assistant(
            messages=messages,
            code=args.code,
            error_message=args.error,
            expected_behavior=args.expected,
            language=args.language,
        )
        messages = result.get("messages", [])
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        if messages and hasattr(messages[-1], "content"):
            print(messages[-1].content)
        print("=" * 60)
    else:
        print("Simple Assistant Workflow CLI (多轮对话模式)")
        print("=" * 60)
        print("输入 exit 或 quit 退出对话")
        print()

        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("再见!")
                break

            messages.append(HumanMessage(content=user_input))

            print("思考中...\n")

            result = run_simple_assistant(
                messages=messages,
                code="",
                error_message="",
                expected_behavior="",
                language="Python",
            )
            messages = result.get("messages", [])

            if messages and hasattr(messages[-1], "content"):
                print(f"\n助手: {messages[-1].content}\n")


if __name__ == "__main__":
    main()
