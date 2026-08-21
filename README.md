This is a Python package to access CMB data from the Legacy Archive for Microwave Background Data (LAMBDA).

## Command line

Installing the package puts a `lambdaquery` command on your PATH:

```bash
lambdaquery experiments                  # list every experiment
lambdaquery datasets WMAP                # list one experiment's datasets
lambdaquery fetch WMAP <dataset> -o ./data
```

`fetch` writes each file under the output directory (`-o`, default: the current
directory), mirroring its LAMBDA `/data/...` path — so the same file referenced by
several datasets is stored, and downloaded, once. Downloaded paths go to stdout and
progress goes to stderr, so `lambdaquery fetch ... > files.txt` gives you a clean list
of paths. Pass `-q` to silence progress. Files already present are skipped.

The listing commands print one name per line and pipe into `grep` — useful, since WMAP
alone has several thousand entries:

```bash
lambdaquery datasets WMAP | grep 9yr
```

`python -m lambdaquery ...` works the same way if the script directory isn't on your PATH.

## Try it

```bash
uv sync
uv run jupyter lab notebooks/lambdaquery_demo.ipynb
```

`notebooks/lambdaquery_demo.ipynb` is a guided tour of the package — browsing the
catalog, inspecting an entry, and downloading a file — with a scratch section at the
end for your own experiments. It's the easiest way to kick the tires and it's where
feedback is most welcome.

For a terminal session instead, `uv run ipython`.
