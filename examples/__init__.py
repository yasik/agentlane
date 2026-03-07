# Make the examples directory into a package to avoid top-level module name collisions.
# This is needed so that mypy treats files like examples/some_example/main.py and
# examples/some_example_2/main.py as distinct modules rather than both named "main".
