# Configuration file for the Sphinx documentation builder.

import os

# --------------------------- Project information ----------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HEASARC Tutorials'
copyright = '2026, HEASARC developers'
author = 'HEASARC developers'
version = '0.1'
# ----------------------------------------------------------------------------

# -------------------------- General configuration ---------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx-specific extensions - the most important here is MyST, and the copybutton extension will also
#  add a small copy button (isn't that shocking) next to code blocks
extensions = ['myst_nb', 'sphinx_copybutton', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.tox', '.tmp', '.pytest_cache', 'README.md',
                    '**/*_template*', '**/README.md', '*_template*']

# Registering custom JS files
#  1. Adds a surface level password-unlocked screen over the website
# THIS HAS BEEN DISABLED, BUT NOT YET REMOVED - "internal_screen.js"
html_js_files = []
# ----------------------------------------------------------------------------

# ---------------------------- MyST configuration ----------------------------
# MyST-NB configuration
nb_execution_timeout = 2400
nb_merge_streams = True
nb_execution_mode = "cache"
nb_scroll_outputs = True

# MyST configurations
# Adding amsmath and dollarmath enables LaTeX style math environments
# The smartquotes extension will automatically convert '' and "" to their nice typeset open and closed versions
# The substitution extension allows us to define keys that will be substituted for a value set in the frontmatter
#  or a centralized file - will be good for defining a value we might want to change everywhere easily
# The colon_fence extension lets us use ::: in place of ``` to delimit directives (I am more used to ::: from using
#  MySTMD)
myst_enable_extensions = ['amsmath', 'dollarmath', 'smartquotes', 'substitution', 'colon_fence']

myst_heading_anchors = 4
# ----------------------------------------------------------------------------

# ------------------ MyST notebook execution configuration -------------------
# Here we define which notebooks are to be executed during the current
#  documentation build. Rather than just specifying the notebooks to execute,
#  we instead must define which SHOULDN'T be executed.
def check_poss_nb(file_name):
    # This could be greatly improved by attempting to read the file frontmatter
    #  using jupytext and marking those files for which an error occurs as not
    #  a notebook.
    return file_name.endswith('.md') and not any([pat in file_name
                                                  for pat in BASE_EXCLUDE_PATTERNS])


# These patterns will always be excluded
BASE_EXCLUDE_PATTERNS = ['*notebook_template*', '*pull_request_template*', '*README*',
                         '**/*README*', '*.ipynb_checkpoints*']

# We allow a 'HEASARC_NOTEBOOKS_TO_BUILD' environment variable to be set, which should
#  have the form:
#  export HEASARC_NOTEBOOKS_TO_BUILD=mission_specific_analyses/swift/getting-started-swift-xrt.md,mission_specific_analyses/nustar/data-analysis-nustar.md

# If the 'HEASARC_NOTEBOOKS_TO_BUILD' environment variable is set, this will read
#  the value and split it into a list of entries. If the variable is not set, the
#  return from 'getenv' is set to be '', and splitting will produce an empty list.
execution_allow_list = os.getenv('HEASARC_NOTEBOOKS_TO_BUILD', '').split(',')
# A little post-processing, to be safe.
# There should not be any spaces in the list entries - it is possible that someone
#  will define the environment variable with comma separation and a space after
#  each comma, and this check makes no appreciable difference to execution speed.
execution_allow_list = [nb_patt.replace(" ", "") for nb_patt in execution_allow_list]
# There should also not be any empty entries in the list
execution_allow_list = [nb_patt for nb_patt in execution_allow_list if nb_patt != ""]

# The 'execution_allow_list' has to be inverted now, as we're excluding all notebooks
#  EXCEPT those in the list.
# If the 'execution_allow_list' is empty, we will default to executing all notebooks.
if len(execution_allow_list) == 0:
    execution_disallow_list = []
# If there ARE entries in the allowed list, then the 'execution_allow_list' has to be
#  inverted, as we're excluding all notebooks EXCEPT those in the list.
else:
    # Start with an empty list of disallowed notebooks
    execution_disallow_list = []

    # Iterate through the generator of the file tree set up by os.walk. This will
    #  take us through all files in all sub-directories of the 'tutorials' directory.
    # Each file will be checked against the allowed list and excluded if it does not
    #  match. Index files in the directories with allowed notebooks will be included
    #  in the build
    for cur_root_dir, cur_dir_names, cur_file_names in os.walk('tutorials'):

        # We need to keep track of which index files we want to includ in the
        #  build, so that on the second iteration through the generator we can
        #  exclude any that don't match
        index_allow_list = []
        for cur_file in cur_file_names:
            # We'll deal with the index files after we've checked the notebooks
            if 'index' in cur_file:
                continue

            # Start by assuming the current file will not be included
            cur_file_include = False
            # Current relative file path
            rel_file_path = os.path.relpath(os.path.join(cur_root_dir, cur_file), 'tutorials')
            # Further checks occur if the file might be a notebook
            if check_poss_nb(cur_file):
                if any([pattern in rel_file_path for pattern in execution_allow_list]):
                    cur_file_include = True

                    # Find the index file in the current directory and add it to
                    #  the list we need to INCLUDE in the build
                    poss_index = [en for en in os.listdir(cur_root_dir) if 'index' in en]
                    # Validity checks - we're okay if there is no index file, but
                    #  there shouldn't be more than one
                    if len(poss_index) == 0:
                        pass
                    elif len(poss_index) > 1:
                        raise ValueError(f"More than one index file found in {cur_root_dir}.")
                    else:
                        rel_index_path = os.path.relpath(os.path.join(cur_root_dir, poss_index[0]), 'tutorials')
                        index_allow_list.append(rel_index_path)

            if not cur_file_include:
                execution_disallow_list.append(rel_file_path)

        # Now we go through the tree-walk again and exclude any index files that weren't
        #  included in the allowed list
        for cur_file in cur_file_names:
            rel_file_path = os.path.relpath(os.path.join(cur_root_dir, cur_file), 'tutorials')

            if 'index' in cur_file and rel_file_path not in index_allow_list:
                execution_disallow_list.append(rel_file_path)

# The final excluded patterns list is the combination of the 'BASE_EXCLUDE_PATTERNS'
#  constant and the disallowed list of notebooks we've just constructed.
nb_execution_excludepatterns = BASE_EXCLUDE_PATTERNS + execution_disallow_list
# ----------------------------------------------------------------------------


# -------------------------- Configure HTML output ---------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'
html_title = 'HEASARC Tutorial Notebooks'
html_logo = '_static/heasarc_logo.png'
html_favicon = '_static/heasarc_favicon.ico'
html_theme_options = {
    "github_url": "https://github.com/HEASARC/heasarc-tutorials",
    "repository_url": "https://github.com/HEASARC/heasarc-tutorials",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "logo": {
        "link": "",
        "alt_text": "High Energy Astrophysics Science Archive Research Center - Home",
        "text": f"v{version}",
    },
    "home_page_in_toc": False,
    "announcement": "<p class='beta-banner'>The HEASARC tutorials resource is in <strong>BETA</strong> and may be subject to significant changes.</p>"
}

# We only want the analytics to be enabled for the production build and website, otherwise
#  whatever data we collect from them will be tainted by our looking at local versions
#  of the website or test builds on CircleCI.
# The GHA that builds the production website will set this environment variable to 'true'
if 'HEASARC_TUTORIALS_ENABLE_ANALYTICS' in os.environ:
    enable_analytics = bool(os.environ['HEASARC_TUTORIALS_ENABLE_ANALYTICS'])
else:
    enable_analytics = False

# If analytics are enabled, add the Google Analytics ID to the theme options
if enable_analytics:
    html_theme_options["analytics"] = {
        "google_analytics_id": "G-R7YGYK7HYQ"
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
# ----------------------------------------------------------------------------
