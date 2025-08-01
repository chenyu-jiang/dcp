from packaging.version import Version as PkgVersion
from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

_flash_attn_min_version = PkgVersion("2.6.3")

ext_modules = [
    Pybind11Extension(
        "dcp_cpp",
        [
            "csrc/pipeline_generator/pipeline_generator.cpp",
            "csrc/interface.cpp",
        ],
        include_dirs=["csrc/pipeline_generator"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", __version__)],
        # extra_compile_args=["-g"],
    ),
]

setup(
    name="dcp",
    version="0.0.1",
    description="Block scheduling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "kahypar",
        "mtkahypar",
        "pybind11",
        "numpy",
        f"flash-attn>={_flash_attn_min_version}",  # higher versions breaks te dependency
        "ring-flash-attn",
        "termcolor",
        "tqdm",
        "redis",
        "transformers",
        "ortools",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
