import typer
from cmpt491_als.modeling.train import app as train_app
from cmpt491_als.modeling.evaluate import app as eval_app

app = typer.Typer()

app.add_typer(train_app, name="fit")
app.add_typer(eval_app, name="eval")

if __name__ == "__main__":
    app()