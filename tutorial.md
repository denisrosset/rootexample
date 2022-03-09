# Handling and documenting data in Python

In this tutorial, we'll see:

- How to document functions/methods/arguments

- How to make a codebase incrementally more robust using types

- Immutability for robustness, and how to document it

- Data classes instead of dicts

- Parsing command line arguments


We'll use a pedagogical implementation of [Brent's root-finding algorithm](https://en.wikipedia.org/wiki/Brent%27s_method) as an example for this tutorial.

We'll start with an implementation that looks like generic code that I'm working on, and we'll make it incrementally more robust.

The files `step1.py` ... `step4.py` are for you to modify. The numbers correspond to the state of
the code *before* the changes are applied.

## Documenting functions/methods/arguments

See the example https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Documenting class attributes

```python
@dataclass
class Example:
    """
    Triple-quoted docstring

    This is standard Python
    """

    a: int #: my super documentation string in autodoc style, not exactly standard

    #: if the line is longer, put the documentation string before
    b: int
```

Luckily, this is already done in `step1.py`.

## Step 1: add types

Install mypy using either `pip install mypy` or `poetry add --dev mypy`.

Add this to `.vscode/settings.json`:

```json
    "python.linting.mypyEnabled": true,
```

You could also add the Pylance checks:
```json
    "python.analysis.diagnosticMode": "workspace",
    "python.analysis.typeCheckingMode": "basic",
```

For types:

Main documentation: https://docs.python.org/3/library/typing.html

Introductory tutorial: https://www.python.org/dev/peps/pep-0483/

We'll use the following types:

- primitive types: `str`, `float`
- `Any` and `Callable`
- data structures: `Dict`, `Tuple`, but see also `List` for completeness

## Step 2: make data structures immutable

There is a bug when displaying the initial point. Why?

Immutable data structures = are not modified. Help you reason about the code!

- `Dict` -> `Mapping`
- `List` -> `Sequence`

## Step 3: use dataclasses

See https://docs.python.org/3/library/dataclasses.html

Replace the `settings` dictionary by a `Settings` class and the `state` dictionary by a `BrentState` class.

Document the attributes.

## Step 4: write a command line application with parameter parsing

Demonstration: run `run_root_cli.sh` (you may need to do `chmod +x run_root_cli.sh`)