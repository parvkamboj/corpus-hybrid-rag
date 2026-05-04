import typer

from cli import eval as eval_module

app = typer.Typer(
    name="corpus",
    help="Corpus RAG — admin and evaluation CLI",
    no_args_is_help=True,
)

app.add_typer(eval_module.app, name="eval")


@app.callback()
def main() -> None:
    pass


if __name__ == "__main__":
    app()
