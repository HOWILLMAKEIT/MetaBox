# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


project = 'Metabox'
copyright = '2025, GMC-Team'
author = 'GMC-Team'
release = 'v0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autodoc2',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinx_copybutton',
]
autodoc2_packages = [
    {
        "path": "../../src",
        # "auto_mode": False,
    },
]
# 设置autodoc2输出Markdown格式
autodoc2_output_format = "myst"

# 确保输出目录存在并已配置
autodoc2_output_dir = "apidocs"

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "myst"),
    # # this will render select docstrings as Markdown
    # (r"autodoc2\..*", "myst"),
]
autodoc2_render_plugin = "myst"
# autodoc2_module_all_regexes = [
#     r"src",
    
# ]

templates_path = ['_templates']
exclude_patterns = []

# 添加 MyST 解析器配置
myst_enable_extensions = [
    "colon_fence"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
# html_theme = 'sphinx_book_theme'
# html_theme = 'press'
html_static_path = ['_static']
locale_dirs = ["locale/"]
gettext_compact = "docs"