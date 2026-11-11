import json
import os
import shutil
import glob
import argparse

def restore(specs, to_build, force_reexecute, state_cache_dir, workspace_root):
    """Surgically restores clean dependency data from state_cache to workspace."""
    missing_data_notebooks = set()
    if not os.path.exists(state_cache_dir):
        print(f"No state cache directory found at {state_cache_dir}")
        # If the cache directory is missing entirely, all notebooks that have cache specs
        #  should be flagged as potentially needing re-execution if they're not fully restored.
        for nb in specs:
            missing_data_notebooks.add(nb)
        return missing_data_notebooks

    for nb, paths in specs.items():
        # All paths are relative to the 'tutorials/' directory
        nb_dir = os.path.dirname(os.path.join('tutorials', nb))
        nb_data_found = False

        # We check for cached data regardless of whether it's being built or not
        #  so we can detect if the cache is actually empty/missing for this notebook.
        for path in paths:
            # The cache stores files at their original relative paths inside 'state_cache'
            pattern = os.path.join(state_cache_dir, nb_dir, path)
            matches = glob.glob(pattern)

            if not matches:
                continue

            nb_data_found = True
            # Only restore data if the notebook is NOT being re-executed in this run
            # We use force_reexecute list instead of to_build here.
            if nb not in force_reexecute:
                for src in matches:
                    # Calculate the destination relative to the repo root
                    dest = os.path.relpath(src, state_cache_dir)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)

                    print(f"Restoring {dest} from cache...")
                    if os.path.isdir(src):
                        if os.path.exists(dest):
                            shutil.rmtree(dest)
                        shutil.copytree(src, dest)
                    else:
                        shutil.copy2(src, dest)

        if not nb_data_found and paths:
            print(f"No cached data found for {nb} - adding to missing list.")
            missing_data_notebooks.add(nb)
        elif nb in force_reexecute:
            print(f"Skipping workspace restoration for {nb} (marked for re-execution), but data was found in cache.")

    return missing_data_notebooks

def collect(specs, to_build, state_cache_dir, workspace_root):
    """Surgically updates state_cache with fresh data from re-executed notebooks."""
    os.makedirs(state_cache_dir, exist_ok=True)

    for nb, paths in specs.items():
        if nb in to_build:
            # Notebook was re-executed. We MUST update the cache with fresh results.
            nb_dir = os.path.dirname(os.path.join('tutorials', nb))
            for path in paths:
                pattern = os.path.join(nb_dir, path)
                matches = glob.glob(pattern)

                if not matches:
                    print(f"No fresh files found matching {pattern} for re-executed notebook {nb}")
                    continue

                for src in matches:
                    # The destination is the same path but inside state_cache
                    dest = os.path.join(state_cache_dir, src)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)

                    print(f"Caching fresh state for {src}...")
                    if os.path.isdir(src):
                        if os.path.exists(dest):
                            shutil.rmtree(dest)
                        shutil.copytree(src, dest)
                    else:
                        shutil.copy2(src, dest)
        else:
            # Notebook was NOT re-executed. Its state in 'state_cache' is already
            # correct (preserved from the previous run).
            print(f"Preserving existing cached state for clean notebook {nb}.")

def main():
    parser = argparse.ArgumentParser(description="Manage sequential notebook data caching.")
    parser.add_argument("--mode", choices=["restore", "collect"], required=True, help="Restore from or collect to cache.")
    parser.add_argument("--cache-dir", default="state_cache", help="Path to the state cache directory.")

    args = parser.parse_args()

    # Read environment variables passed by CircleCI
    try:
        specs = json.loads(os.environ.get('CACHE_SPECS', '{}'))
        to_build = os.environ.get('NOTEBOOKS_TO_BUILD', '').split(',')
        # Clean up empty entries from split
        to_build = [b for b in to_build if b]

        force_reexecute = os.environ.get('FORCE_REEXECUTE', '').split(',')
        force_reexecute = [b for b in force_reexecute if b]
    except Exception as e:
        print(f"Error parsing environment variables: {e}")
        return

    workspace_root = os.getcwd()

    if args.mode == "restore":
        missing = restore(specs, to_build, force_reexecute, args.cache_dir, workspace_root)
        if missing:
            # Output the list of notebooks with missing data to a file that CircleCI can read
            with open("missing_data_notebooks.txt", "w") as f:
                f.write(",".join(missing))
            print(f"Notebooks with missing data: {','.join(missing)}")
    elif args.mode == "collect":
        collect(specs, to_build, args.cache_dir, workspace_root)

if __name__ == "__main__":
    main()
