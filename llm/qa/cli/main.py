import click

from llm.qa.cli.data import data_cli
from llm.qa.cli.pubmed import pubmed_cli


@click.group()
def cli():
    pass


cli.add_command(data_cli)
cli.add_command(pubmed_cli)

if __name__ == "__main__":
    cli()
