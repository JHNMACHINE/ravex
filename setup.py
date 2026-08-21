"""Here for exactly one file.

Everything else about this package is declared in ``pyproject.toml``. This
exists because ``ravex_autoload.pth`` has to land in **purelib** — the
``site-packages`` directory — and setuptools has no declarative way to say so.

The trap is ``data_files``, which looks like the obvious answer and is not:
it puts a file in the wheel's ``<name>-<version>.data/data/`` directory, and
pip installs that relative to ``sys.prefix``. In a virtualenv that is the
environment root, one level above ``site-packages``. The file arrives, looks
installed, and is never executed — because Python only runs ``.pth`` files
found in a site directory.

Files at the **root of the wheel archive** go to purelib. So the build puts it
there itself.

Because it ends up in the wheel it also ends up in ``RECORD``, which is what
makes ``pip uninstall ravex`` take it away again. That matters more than it
looks: an orphaned ``.pth`` importing a module that no longer exists is the
classic way this scheme breaks, and it is not a risk here.
"""

import os

from setuptools import setup
from setuptools.command.build_py import build_py

AUTOLOADER = "ravex_autoload.pth"


class BuildWithAutoloader(build_py):
    """``build_py``, plus the one file that has to sit beside the package."""

    def run(self):
        super().run()
        if self.dry_run:
            return
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), AUTOLOADER)
        self.copy_file(source, os.path.join(self.build_lib, AUTOLOADER))

    def get_outputs(self, include_bytecode=1):
        # Listed so the commands that ask what this build produced — and the
        # installer that records it for uninstall — are told about it.
        outputs = super().get_outputs(include_bytecode)
        return [*outputs, os.path.join(self.build_lib, AUTOLOADER)]


setup(cmdclass={"build_py": BuildWithAutoloader})
