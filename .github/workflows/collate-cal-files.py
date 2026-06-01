import jupytext
import os

if __name__ == '__main__':

    # The location of the checked out HEASARC-tutorials repository
    cur_workspace = os.environ['GITHUB_WORKSPACE']

    # Location of the template notebook - it contains the names of all current calibration
    #  data sets that the execution workflow can download and cache
    cur_nb_temp_path = os.path.join(cur_workspace, 'notebook-templates', 'notebook_template.md')
    # Load the file and extract the allowed names from the metadata (i.e. front-matter)
    with open(cur_nb_temp_path, 'r') as md_read:
        cur_nb_temp_jupy = jupytext.reads(md_read.read())
        allowed_cal_names = list(cur_nb_temp_jupy.metadata['execution']['cal-files'].keys())

    # Paths to notebooks that will be executed in this GHA run, relative to the
    #  top-level repository directory. This could either be a single notebook, or a
    #  set of notebooks separated by ','
    rel_nbs = os.environ['NOTEBOOKS_TO_BUILD']
    # Make sure that the rel_nbs variable is a Python list of notebooks (will become
    #  a one-element list if there is only one notebook).
    rel_nbs = rel_nbs.split(',')

    # The bucket we'll add all the notebook-author-specified calibration sets to, there may well be
    #  duplicates by the end, but we'll take care of that.
    all_cal_files = []
    # Then a dictionary in which we can record any illegal entries in the execution: cal-files: section
    #  of the front matter. That will get passed back out to the GHA, which will then raise an error and
    #  fail the action
    bad_meta_spec = {}

    # Iterate through the notebooks that we'll be executing
    for cur_nb in rel_nbs:
        # Define the full path
        cur_nb_path = os.path.join(cur_workspace, cur_nb)

        # Load the file and parse with Jupytext
        with open(cur_nb_path, 'r') as md_read:
            cur_nb_jupy = jupytext.reads(md_read.read())

        # From here, accessing the front-matter information is very easy, as it has
        #  already been stored in the 'metadata' property.
        # We do check that there is an entry for 'cal-files' - this will become part of
        #  the HEASARC-tutorials MD standard/format, but best to be flexible regarding
        #  whether the information is there or not for now.
        # If no information is available, we assume all calibration sets are required
        if 'execution' not in cur_nb_jupy.metadata or 'cal-files' not in cur_nb_jupy.metadata['execution']:
            all_cal_files += allowed_cal_names
            continue
        elif len(bad_cal := [cur_cal_name for cur_cal_name in cur_nb_jupy.metadata['execution']['cal-files'].keys() if cur_cal_name not in allowed_cal_names]) != 0:
            bad_meta_spec[cur_nb] = bad_cal
            continue
        elif len(bad_cal_val := [f"{cur_cal_name} ({cur_cal_val})" for cur_cal_name, cur_cal_val in cur_nb_jupy.metadata['execution']['cal-files'].items() if not isinstance(cur_cal_val, bool)]) != 0:
            bad_meta_spec[cur_nb] = bad_cal_val
            continue

        # If we get here, then calibration set information WAS included in the notebook front-matter, and as far as
        #  we can tell, it was formatted correctly. All we have left to do is sweep through the cal-files and see
        #  which ones were marked as True
        cur_sel_cal = [cur_cal_name for cur_cal_name, cur_cal_val in cur_nb_jupy.metadata['execution']['cal-files'].items() if cur_cal_val]
        # Put the selected calibration sets in the bucket we're using for every notebook that has
        #  been marked for building using the CircleCI workflow.
        all_cal_files += cur_sel_cal

    # At this point we've run through all the notebooks, so we can find the unique entries in
    #  the all_cal_files variable we populated in the loop, and those are then the calibration
    #  data sets that we need to tell the CircleCI workflow to load.
    all_cal_files = list(set(all_cal_files))

    # Now we write to the GitHub actions output file for the current step
    cur_gha_out = os.environ["GITHUB_OUTPUT"]

    with open(cur_gha_out, "a") as gha_outo:
        # This writes the names of the selected calibration sets to a file that will be
        #  read and accessible in the next GHA step (which will be the part where
        #  the CircleCI workflow is actually triggered)
        gha_outo.write(f"required_cal_sets={",".join(all_cal_files)}\n")

        # In this case, some illegal entry was identified in one or more of the notebooks
        #  specified for building, and we need to make sure that information is available
        #  to the next step so it can error out
        if len(bad_meta_spec) != 0:
            # Tells the next step right off the bat that something is wrong
            gha_outo.write("any_bad_cal_spec=true\n")
            # Then this information will be shown to the notebook author so they can
            #  diagnose what went wrong. We don't prettify the output, as it doesn't
            #  really need to be pretty
            gha_outo.write(str(bad_meta_spec) + "\n")
        else:
            # We always write the 'any_bad_cal_spec entry, so the next GHA step can check to
            #  see if it's True before having to parse the actual information about what went wrong
            gha_outo.write("any_bad_cal_spec=false\n")
