from __future__ import annotations

import argparse
import os
from pathlib import Path

from quiz_agent import SMTPConfig, agent_mode_send_quiz, generate_quiz_from_pdf


def _positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return iv


def cmd_generate(args: argparse.Namespace) -> None:
    quiz = generate_quiz_from_pdf(
        pdf_path=args.pdf,
        num_questions=args.num_questions,
        seed=args.seed,
    )
    if args.out:
        Path(args.out).write_text(quiz, encoding="utf-8")
    else:
        print(quiz)


def cmd_send(args: argparse.Namespace) -> None:
    host = args.host or os.environ.get("SMTP_HOST")
    port_str = args.port or os.environ.get("SMTP_PORT")
    username = args.username or os.environ.get("SMTP_USERNAME")
    password = args.password or os.environ.get("SMTP_PASSWORD")
    use_tls = not args.no_tls if args.no_tls is not None else os.environ.get("SMTP_USE_TLS", "true").lower() != "false"

    if not host or not port_str or not username or not password:
        raise SystemExit("Missing SMTP settings. Provide via flags or env: SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD")

    port = int(port_str)
    smtp_config = SMTPConfig(host=host, port=port, username=username, password=password, use_tls=use_tls)

    agent_mode_send_quiz(
        pdf_path=args.pdf,
        to_email=args.to,
        subject=args.subject,
        smtp_config=smtp_config,
        num_questions=args.num_questions,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a quiz from a PDF and optionally email it.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate", help="Generate quiz text from PDF")
    p_gen.add_argument("pdf", help="Path to input PDF")
    p_gen.add_argument("--num-questions", type=_positive_int, default=5)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--out", help="Path to save quiz text; stdout if omitted")
    p_gen.set_defaults(func=cmd_generate)

    p_send = subparsers.add_parser("send", help="Generate quiz and send via SMTP")
    p_send.add_argument("pdf", help="Path to input PDF")
    p_send.add_argument("to", help="Recipient email address")
    p_send.add_argument("subject", help="Email subject")
    p_send.add_argument("--num-questions", type=_positive_int, default=5)
    p_send.add_argument("--seed", type=int, default=None)
    p_send.add_argument("--host", help="SMTP host (or env SMTP_HOST)")
    p_send.add_argument("--port", help="SMTP port (or env SMTP_PORT)")
    p_send.add_argument("--username", help="SMTP username (or env SMTP_USERNAME)")
    p_send.add_argument("--password", help="SMTP password (or env SMTP_PASSWORD)")
    p_send.add_argument("--no-tls", action="store_true", help="Disable STARTTLS and use SSL instead")
    p_send.set_defaults(func=cmd_send)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

