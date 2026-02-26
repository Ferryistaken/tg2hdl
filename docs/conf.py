from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'tg2hdl'
author = 'tg2hdl contributors'
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'node_modules', '.vitepress']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

html_theme = 'furo'
html_title = 'tg2hdl docs'
html_static_path = ['_static']

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

html_css_files = ['custom.css']
