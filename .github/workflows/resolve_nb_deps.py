import os
import yaml
import sys
import argparse
from typing import List, Dict, Set
import re

def parse_metadata(file_path: str) -> dict:
    """Parses YAML frontmatter from a Jupytext markdown file."""
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Jupytext frontmatter is between --- markers
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}

    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}

def get_allowed_cal_names(root_dir: str) -> List[str]:
    """Reads the allowed calibration names from the notebook template."""
    template_path = os.path.join(os.path.dirname(root_dir), 'notebook-templates', 'notebook_template.md')
    if not os.path.exists(template_path):
        # Fallback if template missing
        return ['xmm-ccf', 'chandra', 'xspec-models']

    metadata = parse_metadata(template_path)
    try:
        return list(metadata.get('execution', {}).get('cal-files', {}).keys())
    except Exception:
        return ['xmm-ccf', 'chandra', 'xspec-models']

def build_reverse_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Builds a map of notebooks to their dependents."""
    rev_graph = {node: [] for node in graph}
    for node, deps in graph.items():
        for dep in deps:
            if dep in rev_graph:
                rev_graph[dep].append(node)
    return rev_graph

def build_dependency_graph(root_dir: str) -> Dict[str, List[str]]:
    """Builds a map of notebook paths to their dependencies."""
    graph = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md') and 'notebook_template' not in file:
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                # Skip index files and templates
                if 'index' in file or 'README' in file:
                    continue

                metadata = parse_metadata(os.path.join(root, file))
                exec_meta = metadata.get('execution', {})
                deps = exec_meta.get('dependencies', [])

                # Normalize paths (they are relative to the tutorials directory)
                graph[rel_path] = deps
    return graph

def get_full_execution_set(graph: Dict[str, List[str]], rev_graph: Dict[str, List[str]], targets: List[str]) -> Set[str]:
    """Finds all notebooks in the dependency chains (up and down) of the targets."""
    to_build = set(targets)

    # 1. Add all upstream dependencies (what we need to run first)
    stack = list(targets)
    while stack:
        current = stack.pop()
        deps = graph.get(current, [])
        for dep in deps:
            if dep not in to_build:
                if dep not in graph:
                    print(f"Warning: Dependency {dep} not found in tutorials directory.", file=sys.stderr)
                    continue
                to_build.add(dep)
                stack.append(dep)

    # 2. Add all downstream dependents (what might be broken by changes)
    stack = list(targets)
    while stack:
        current = stack.pop()
        dependents = rev_graph.get(current, [])
        for dep in dependents:
            if dep not in to_build:
                to_build.add(dep)
                stack.append(dep)

    return to_build

def topological_sort(graph: Dict[str, List[str]], subset: Set[str]) -> List[str]:
    """Sorts the subset of notebooks in execution order."""
    result = []
    visited = set()
    temp_visited = set()

    def visit(n):
        if n in temp_visited:
            raise ValueError(f"Circular dependency detected involving {n}")
        if n not in visited:
            temp_visited.add(n)
            for dep in graph.get(n, []):
                if dep in subset:
                    visit(dep)
            temp_visited.remove(n)
            visited.add(n)
            result.append(n)

    for node in sorted(subset):
        if node not in visited:
            visit(node)

    return result

def get_calibration_needs(root_dir: str, notebooks: List[str], allowed_names: List[str]) -> Dict[str, bool]:
    """Aggregates calibration requirements for the set of notebooks."""
    needs = {name: False for name in allowed_names}
    bad_meta = {}

    for nb in notebooks:
        nb_path = os.path.join(root_dir, nb)
        metadata = parse_metadata(nb_path)

        exec_meta = metadata.get('execution', {})

        if not exec_meta or 'cal-files' not in exec_meta:
            # Default to ALL calibration if none specified
            for name in allowed_names:
                needs[name] = True
            continue

        cal_files = exec_meta.get('cal-files', {})

        # Check for invalid names
        invalid = [n for n in cal_files.keys() if n not in allowed_names]
        if invalid:
            bad_meta[nb] = invalid
            continue

        for name, value in cal_files.items():
            if value:
                needs[name] = True

    return needs, bad_meta

def get_cache_specs(root_dir: str, notebooks: List[str]) -> Dict[str, List[str]]:
    """Collects paths to cache for each notebook."""
    specs = {}
    for nb in notebooks:
        metadata = parse_metadata(os.path.join(root_dir, nb))
        exec_meta = metadata.get('execution', {})
        cache_outputs = exec_meta.get('cache_outputs', [])
        if cache_outputs:
            specs[nb] = cache_outputs
    return specs

def main():
    parser = argparse.ArgumentParser(description="Resolve notebook dependencies and calibration needs.")
    parser.add_argument("--changed-files", required=True, help="Comma-separated list of changed .md files relative to tutorials/")
    parser.add_argument("--tutorials-dir", default="tutorials", help="Path to the tutorials directory")

    args = parser.parse_args()

    abs_tutorials_dir = os.path.abspath(args.tutorials_dir)

    # 1. Clean up changed files list
    changed = [f.strip() for f in args.changed_files.split(',') if f.strip().endswith('.md')]

    # Remove 'tutorials/' prefix if present
    changed = [f[len('tutorials/'):] if f.startswith('tutorials/') else f for f in changed]

    if not changed:
        print("NOTEBOOKS_TO_BUILD=")
        print("ANY_BAD_CAL_SPEC=false")
        print("REQUIRE_XMM_CCF=false")
        print("REQUIRE_CHANDRA=false")
        print("REQUIRE_XSPEC_MODELS=false")
        return

    # 2. Build graphs and find all required notebooks
    try:
        graph = build_dependency_graph(args.tutorials_dir)
        rev_graph = build_reverse_graph(graph)
        allowed_cal = get_allowed_cal_names(args.tutorials_dir)

        # Filter changed files to only those inside the graph (ignores templates, etc.)
        targets = [f for f in changed if f in graph]

        # Get closure
        full_set = get_full_execution_set(graph, rev_graph, targets)

        # Sort
        ordered_list = topological_sort(graph, full_set)

        # 3. Determine calibration needs
        cal_needs, bad_cal = get_calibration_needs(args.tutorials_dir, ordered_list, allowed_cal)

        # 4. Determine cache specs
        cache_specs = get_cache_specs(args.tutorials_dir, ordered_list)

        # 5. Output results in a format GitHub Actions can parse
        print(f"NOTEBOOKS_TO_BUILD={','.join(ordered_list)}")

        if bad_cal:
            print("ANY_BAD_CAL_SPEC=true")
            print(f"BAD_CAL_SPEC={str(bad_cal)}")
        else:
            print("ANY_BAD_CAL_SPEC=false")

        # Write individual calibration flags
        for name in allowed_cal:
            # CircleCI parameters use underscores, not dashes
            param_name = name.replace('-', '_').upper()
            print(f"REQUIRE_{param_name}={str(cal_needs[name]).lower()}")

        # Export cache specs as a JSON string for the build script
        import json
        print(f"CACHE_SPECS={json.dumps(cache_specs)}")

    except Exception as e:
        print(f"Error resolving dependencies: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
