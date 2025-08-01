import os
import argparse

os.environ["DCP_DEBUG"] = "DEBUG"
os.environ["DCP_DEBUG_LOAD_PARTITION_RESULTS"] = "1"

################## Set this to the debug dump #################################
os.environ["DCP_DEBUG_LOAD_PARTITION_RESULTS_FILE"] = None


from dcp.core.compiler import InstrCompiler, CompilerConfig
from dcp.core.common import get_default_execution_context

# ################## Modify number of devices and nodes #######################
context = get_default_execution_context(8 * 4, 4)
config = CompilerConfig()
compiler = InstrCompiler(context, config, 0)
compiler.compile(None)
