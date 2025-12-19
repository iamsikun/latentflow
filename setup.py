from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="latentflow.core._hmm_cy",
        sources=["src/latentflow/core/_hmm_cy.pyx"],
        include_dirs=[np.get_include()],
    )
]

from pathlib import Path

# Fix for absolute paths in extensions (required for setuptools validation)
ext_modules = cythonize(
    extensions,
    language_level="3",
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
    },
)

for ext in ext_modules:
    new_sources = []
    for source in ext.sources:
        source_path = Path(source)
        if source_path.is_absolute():
            try:
                # Try to make it relative to the current working directory
                relative_source = source_path.relative_to(Path.cwd())
                new_sources.append(str(relative_source))
            except ValueError:
                # Keep original if it can't be made relative
                new_sources.append(source)
        else:
            new_sources.append(source)
    ext.sources = new_sources

setup(ext_modules=ext_modules)
