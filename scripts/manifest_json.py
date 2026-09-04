#!/usr/bin/env python3
"""Write the lambdaquery manifest as canonical JSON.

Standalone by design: this is meant to be copied to the LAMBDA server, where
`lambdaquery` is not installed. `dump_manifest` and `validate` need nothing but
the standard library; PyYAML is imported lazily and only by the YAML->JSON
conversion path, so a generator that already builds the dict in memory can
`from manifest_json import dump_manifest` and skip YAML entirely.

Deliberately conservative about Python version (no walrus, no `X | None`
annotations, no dataclasses) so it runs on an older server interpreter.

Usage:
    python3 manifest_json.py manifest.yaml manifest.json   # convert
    python3 manifest_json.py --check manifest.json         # validate only

Or from your generator:
    from manifest_json import dump_manifest, validate
    errors = validate(data)
    if not errors:
        dump_manifest(data, "manifest.json")
"""

import argparse
import json
import os
import sys

# Mirrors Registry._LEAF_KEYS / _GROUP_KEYS in src/lambdaquery/_registry.py.
# Kept in sync by hand; the authoritative check is loading the result with
# Registry, which `--check` approximates for a machine that lacks the package.
LEAF_KEYS = frozenset(["path", "size", "checksum", "description"])
GROUP_KEYS = frozenset(["children", "description"])


def dump_manifest(data, path):
    """Write ``data`` to ``path`` as canonical manifest JSON.

    indent=1 keeps one entry per line (greppable, diffable) for ~3% over
    minified; sort_keys makes regeneration diffs show only real changes, and is
    safe because the library sorts experiment/entry names on read and sort_keys
    never reorders a list, so group `children` keep their download order.
    ensure_ascii=False keeps any non-ASCII in a description readable, which is
    why the file must be opened with an explicit encoding. The write goes to a
    temp file and is renamed into place, so a crash mid-write cannot leave a
    truncated manifest behind -- the same pattern _fetch uses for downloads.
    """
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def validate(data):
    """Return a list of human-readable problems with a manifest dict.

    Mirrors what Registry._build_entry enforces at load time -- entry shape,
    mapping keys, field types, dangling references and reference cycles -- so a
    bad manifest is caught on the machine that generated it rather than by
    breaking `import lambdaquery` after it ships.
    """
    errors = []
    if not isinstance(data, dict):
        return ["top level must be a mapping of experiment -> entries"]

    for exp in sorted(data):
        entries = data[exp]
        if not isinstance(entries, dict):
            errors.append("%s: must map entry names to entries" % exp)
            continue
        for name in sorted(entries):
            _check_entry(exp, entries, name, (), errors)
    return errors


def _check_entry(exp, entries, name, chain, errors):
    where = "%s/%s" % (exp, name)
    if name in chain:
        errors.append("%s: reference cycle %s" % (where, " -> ".join(chain + (name,))))
        return
    if name not in entries:
        errors.append("%s: referenced but not defined" % where)
        return

    value = entries[name]
    if isinstance(value, str):
        return
    if isinstance(value, list):
        _check_children(exp, entries, name, value, chain, errors)
        return
    if not isinstance(value, dict):
        errors.append(
            "%s: must be a path string, a list of names, or a mapping "
            "(got %s)" % (where, type(value).__name__)
        )
        return

    has_path = "path" in value
    has_children = "children" in value
    if has_path and has_children:
        errors.append("%s: sets both 'path' and 'children'" % where)
        return
    if not has_path and not has_children:
        errors.append(
            "%s: sets neither 'path' nor 'children' (keys: %s)" % (where, sorted(value))
        )
        return

    allowed = LEAF_KEYS if has_path else GROUP_KEYS
    for key in sorted(set(value) - set(allowed)):
        errors.append(
            "%s: unknown key %r (allowed: %s)" % (where, key, sorted(allowed))
        )

    if "description" in value and not isinstance(value["description"], str):
        errors.append("%s: 'description' must be a string" % where)

    if has_children:
        children = value["children"]
        if not isinstance(children, list):
            errors.append("%s: 'children' must be a list of entry names" % where)
        else:
            _check_children(exp, entries, name, children, chain, errors)
        return

    if not isinstance(value["path"], str):
        errors.append("%s: 'path' must be a string" % where)
    if "checksum" in value and not isinstance(value["checksum"], str):
        errors.append("%s: 'checksum' must be a string" % where)
    if "size" in value:
        size = value["size"]
        # bool is an int subclass; "size: true" is a mistake, not a size.
        if not isinstance(size, int) or isinstance(size, bool):
            errors.append(
                "%s: 'size' must be an integer number of bytes (got %s)"
                % (where, type(size).__name__)
            )
        elif size < 0:
            errors.append("%s: 'size' must not be negative" % where)


def _check_children(exp, entries, name, children, chain, errors):
    for child in children:
        if not isinstance(child, str):
            errors.append(
                "%s/%s: child names must be strings (got %s)"
                % (exp, name, type(child).__name__)
            )
            continue
        _check_entry(exp, entries, child, chain + (name,), errors)


def coverage(data):
    """Return (leaves, with_size, with_checksum, groups) across the manifest."""
    leaves = sized = summed = groups = 0
    for entries in data.values():
        if not isinstance(entries, dict):
            continue
        for value in entries.values():
            if isinstance(value, str):
                leaves += 1
            elif isinstance(value, list):
                groups += 1
            elif isinstance(value, dict):
                if "path" in value:
                    leaves += 1
                    sized += "size" in value
                    summed += "checksum" in value
                else:
                    groups += 1
    return leaves, sized, summed, groups


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit(
            "PyYAML is required to read a YAML manifest.\n"
            "  pip install --user pyyaml   (or convert with --check on JSON only)"
        )
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=loader)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="manifest to read (.yaml or .json)")
    p.add_argument("dst", nargs="?", help="JSON file to write; omit with --check")
    p.add_argument(
        "--check",
        action="store_true",
        help="validate only, write nothing",
    )
    args = p.parse_args(argv)

    if args.src.endswith((".yaml", ".yml")):
        data = load_yaml(args.src)
    else:
        with open(args.src, "r", encoding="utf-8") as f:
            data = json.load(f)

    errors = validate(data)
    leaves, sized, summed, groups = coverage(data)
    total = leaves + groups
    sys.stderr.write(
        "%s: %d entries (%d file leaves, %d groups)\n"
        "  size:     %d/%d leaves (%.1f%%)\n"
        "  checksum: %d/%d leaves (%.1f%%)\n"
        % (
            args.src,
            total,
            leaves,
            groups,
            sized,
            leaves,
            100.0 * sized / leaves if leaves else 0.0,
            summed,
            leaves,
            100.0 * summed / leaves if leaves else 0.0,
        )
    )

    if errors:
        sys.stderr.write("\n%d problem(s):\n" % len(errors))
        for e in errors[:50]:
            sys.stderr.write("  %s\n" % e)
        if len(errors) > 50:
            sys.stderr.write("  ... and %d more\n" % (len(errors) - 50))
        return 1

    sys.stderr.write("  manifest is structurally valid\n")

    if args.check:
        return 0
    if not args.dst:
        p.error("a destination is required unless --check is given")

    dump_manifest(data, args.dst)
    sys.stderr.write("  wrote %s (%d bytes)\n" % (args.dst, os.path.getsize(args.dst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
