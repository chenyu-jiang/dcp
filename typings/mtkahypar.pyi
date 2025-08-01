from __future__ import annotations
import typing

__all__ = [
    "Context",
    "FileFormat",
    "Graph",
    "Hypergraph",
    "Initializer",
    "InvalidInputError",
    "InvalidParameterError",
    "Objective",
    "PartitionedHypergraph",
    "PresetType",
    "SystemError",
    "TargetGraph",
    "UnsupportedOperationError",
    "initialize",
    "set_seed",
]

class Context:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    @staticmethod
    def set_partitioning_parameters(*args, **kwargs) -> None:
        """
        Sets all required parameters for partitioning
        """

    def compute_max_block_weights(self, total_weight: int) -> list[int]:
        """
        Compute maximum allowed block weights for an input with the given total weight (depends on epsilon).
        If indiviual target weights are set, these are returned instead.

        :param total_weight: Total weight of input hypergraph
        """

    def compute_perfect_balance_block_weights(
        self, total_weight: int
    ) -> list[int]:
        """
        Compute block weights that represent perfect balance for an input with the given total weight.
        If indiviual target weights are set, these are used for the calculation instead.

        :param total_weight: Total weight of input hypergraph
        """

    def print_configuration(self) -> None:
        """
        Print partitioning configuration
        """

    def set_individual_target_block_weights(
        self, block_weights: list[int]
    ) -> None:
        """
        Set individual maximum allowed weight for each block of the output partition
        """

    def set_mapping_parameters(self, k: int, epsilon: float) -> None:
        """
        Sets all required parameters for mapping to a target graph
        """

    @property
    def epsilon(self) -> float:
        """
        Allowed imbalance
        """

    @epsilon.setter
    def epsilon(self, arg1: float) -> None: ...
    @property
    def k(self) -> int:
        """
        Number of blocks in which the (hyper)graph should be partitioned into
        """

    @k.setter
    def k(self, arg1: int) -> None: ...
    @property
    def logging(self) -> bool:
        """
        Enable partitioning output
        """

    @logging.setter
    def logging(self, arg1: bool) -> None: ...
    @property
    def num_vcycles(self) -> int:
        """
        Sets the number of V-cycles
        """

    @num_vcycles.setter
    def num_vcycles(self, arg1: int) -> None: ...
    @property
    def objective(self) -> Objective:
        """
        Sets the objective function for partitioning (CUT, KM1 or SOED)
        """

    @objective.setter
    def objective(self, arg1: Objective) -> None: ...
    @property
    def preset(self) -> PresetType:
        """
        Preset of the context
        """

class FileFormat:
    """
    Members:

      HMETIS

      METIS
    """

    HMETIS: typing.ClassVar[FileFormat]  # value = <FileFormat.HMETIS: 0>
    METIS: typing.ClassVar[FileFormat]  # value = <FileFormat.METIS: 1>
    __members__: typing.ClassVar[
        dict[str, FileFormat]
    ]  # value = {'HMETIS': <FileFormat.HMETIS: 0>, 'METIS': <FileFormat.METIS: 1>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Graph(Hypergraph):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def edge_source(self, edge: int) -> int:
        """
        Source node of edge (e.g., (0,1) -> 0 is the source node)
        """

    def edge_target(self, edge: int) -> int:
        """
        Target node of edge (e.g., (0,1) -> 1 is the target node)
        """

    def num_directed_edges(self) -> int:
        """
        Number of directed edges (equal to num_edges)
        """

    def num_undirected_edges(self) -> int:
        """
        Number of undirected edges (equal to num_edges / 2)
        """

class Hypergraph:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def add_fixed_vertices(
        self, fixed_vertices: list[int], num_blocks: int
    ) -> None:
        """
        Adds the fixed vertices specified in the array to the (hyper)graph. The array must contain
        n entries (n = number of nodes). Each entry contains either the fixed vertex block of the
        corresponding node or -1 if the node is not fixed.
        """

    def add_fixed_vertices_from_file(
        self, fixed_vertex_file: str, num_blocks: int
    ) -> None:
        """
        Adds the fixed vertices specified in the fixed vertex file to the (hyper)graph. The file must contain
        n lines (n = number of nodes). Each line contains either the fixed vertex block of the
        corresponding node or -1 if the node is not fixed.
        """

    def create_partitioned_hypergraph(
        self, num_blocks: Context, context: int, partition: list[int]
    ) -> PartitionedHypergraph:
        """
        Construct a partitioned hypergraph from this hypergraph.

        :param num_blocks: number of block in which the hypergraph should be partitioned into
        :param partition: list of block IDs for each node
        """

    def edge_size(self, hyperedge: int) -> int:
        """
        Size of hyperedge
        """

    def edge_weight(self, hyperedge: int) -> int:
        """
        Weight of hyperedge
        """

    def edges(self) -> typing.Iterator:
        """
        Iterator over all hyperedges
        """

    def fixed_vertex_block(self, node: int) -> int:
        """
        Block to which the node is fixed (-1 if not fixed)
        """

    def incident_edges(self, node: int) -> typing.Iterator:
        """
        Iterator over incident hyperedges of node
        """

    def is_compatible(self, preset: PresetType) -> bool:
        """
        Returns whether or not the given hypergraph can be partitioned with the preset
        """

    def is_fixed(self, node: int) -> bool:
        """
        Returns whether or not the corresponding node is a fixed vertex
        """

    def map_onto_graph(
        self, target_graph: TargetGraph, context: Context
    ) -> PartitionedHypergraph:
        """
        Maps a (hyper)graph onto a target graph with the parameters given in the partitioning context.
        The number of blocks of the output mapping/partition is the same as the number of nodes in the target graph
        (each node of the target graph represents a block). The objective is to minimize the total weight of
        all Steiner trees spanned by the (hyper)edges on the target graph. A Steiner tree is a tree with minimal weight
        that spans a subset of the nodes (in our case the hyperedges) on the target graph. This objective function
        is able to acurately model wire-lengths in VLSI design or communication costs in a distributed system where some
        processors do not communicate directly with each other or different speeds.
        """

    def node_degree(self, node: int) -> int:
        """
        Degree of node
        """

    def node_weight(self, node: int) -> int:
        """
        Weight of node
        """

    def nodes(self) -> typing.Iterator:
        """
        Iterator over all nodes
        """

    def num_edges(self) -> int:
        """
        Number of hyperedges
        """

    def num_nodes(self) -> int:
        """
        Number of nodes
        """

    def num_pins(self) -> int:
        """
        Number of pins
        """

    def partition(self, context: Context) -> PartitionedHypergraph:
        """
        Partitions the hypergraph with the parameters given in the partitioning context
        """

    def partitioned_hypergraph_from_file(
        self, num_blocks: Context, context: int, partition_file: str
    ) -> PartitionedHypergraph:
        """
        Construct a partitioned hypergraph from this hypergraph.

        :param num_blocks: number of block in which the hypergraph should be partitioned into
        :param partition_file: partition file containing block IDs for each node
        """

    def pins(self, hyperedge: int) -> typing.Iterator:
        """
        Iterator over pins of hyperedge
        """

    def remove_fixed_vertices(self) -> None:
        """
        Removes all fixed vertices from the hypergraph
        """

    def total_weight(self) -> int:
        """
        Total weight of all nodes
        """

class Initializer:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def context_from_file(self, config_file: str) -> Context:
        """
        Creates a context from a configuration file.
        """

    def context_from_preset(self, preset: PresetType) -> Context:
        """
        Creates a context from the given preset.
        """

    @typing.overload
    def create_graph(
        self,
        context: Context,
        num_nodes: int,
        num_edges: int,
        edges: list[tuple[int, int]],
    ) -> Graph:
        """
        Construct an unweighted graph.

        :param context: the partitioning context
        :param num_nodes: Number of nodes
        :param num_edges: Number of edges
        :param edges: list of tuples containing all edges (e.g., [(0,1),(0,2),(1,3),...])
        """

    @typing.overload
    def create_graph(
        self,
        context: Context,
        num_nodes: int,
        num_edges: int,
        edges: list[tuple[int, int]],
        node_weights: list[int],
        edge_weights: list[int],
    ) -> Graph:
        """
        Construct a weighted graph.

        :param context: the partitioning context
        :param num_nodes: Number of nodes
        :param num_edges: Number of edges
        :param edges: list of tuples containing all edges (e.g., [(0,1),(0,2),(1,3),...])
        :param node_weights: Weights of all nodes
        :param edge_weights: Weights of all edges
        """

    @typing.overload
    def create_hypergraph(
        self,
        context: Context,
        num_hypernodes: int,
        num_hyperedges: int,
        hyperedges: list[list[int]],
    ) -> Hypergraph:
        """
        Construct an unweighted hypergraph.

        :param context: the partitioning context
        :param num_hypernodes: Number of nodes
        :param num_hyperedges: Number of hyperedges
        :param hyperedges: list containing all hyperedges (e.g., [[0,1],[0,2,3],...])
        """

    @typing.overload
    def create_hypergraph(
        self,
        context: Context,
        num_hypernodes: int,
        num_hyperedges: int,
        hyperedges: list[list[int]],
        node_weights: list[int],
        hyperedge_weights: list[int],
    ) -> Hypergraph:
        """
        Construct a weighted hypergraph.

        :param context: the partitioning context
        :param num_hypernodes: Number of nodes
        :param num_hyperedges: Number of hyperedges
        :param hyperedges: List containing all hyperedges (e.g., [[0,1],[0,2,3],...])
        :param node_weights: Weights of all hypernodes
        :param hyperedge_weights: Weights of all hyperedges
        """

    def create_target_graph(
        self,
        context: Context,
        num_nodes: int,
        num_edges: int,
        edges: list[tuple[int, int]],
        edge_weights: list[int],
    ) -> TargetGraph:
        """
        Construct a target graph.

        :param context: the partitioning context
        :param num_nodes: Number of nodes
        :param num_edges: Number of edges
        :param edges: list of tuples containing all edges (e.g., [(0,1),(0,2),(1,3),...])
        :param edge_weights: Weights of all edges
        """

    def graph_from_file(
        self, filename: str, context: Context, format: FileFormat = ...
    ) -> Graph:
        """
        Reads a graph from a file (supported file formats are METIS and HMETIS)
        """

    def hypergraph_from_file(
        self, filename: str, context: Context, format: FileFormat = ...
    ) -> Hypergraph:
        """
        Reads a hypergraph from a file (supported file formats are METIS and HMETIS)
        """

    def target_graph_from_file(
        self, filename: str, context: Context, format: FileFormat = ...
    ) -> TargetGraph:
        """
        Reads a target graph from a file (supported file formats are METIS and HMETIS)
        """

class InvalidInputError(ValueError):
    pass

class InvalidParameterError(ValueError):
    pass

class Objective:
    """
    Members:

      CUT

      KM1

      SOED
    """

    CUT: typing.ClassVar[Objective]  # value = <Objective.CUT: 0>
    KM1: typing.ClassVar[Objective]  # value = <Objective.KM1: 1>
    SOED: typing.ClassVar[Objective]  # value = <Objective.SOED: 2>
    __members__: typing.ClassVar[
        dict[str, Objective]
    ]  # value = {'CUT': <Objective.CUT: 0>, 'KM1': <Objective.KM1: 1>, 'SOED': <Objective.SOED: 2>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PartitionedHypergraph:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def block_id(self, node: int) -> int:
        """
        Block to which the corresponding node is assigned
        """

    def block_weight(self, block: int) -> int:
        """
        Weight of the corresponding block
        """

    def blocks(self) -> typing.Iterator:
        """
        Iterator over blocks of the partition
        """

    def connectivity(self, hyperedge: int) -> int:
        """
        Number of distinct blocks to which the pins of the corresponding hyperedge are assigned
        """

    def connectivity_set(self, hyperedge: int) -> typing.Iterator:
        """
        Iterator over blocks to which the pins of the corresponding hyperedge are assigned
        """

    def cut(self) -> int:
        """
        Computes the cut-net metric of the partition
        """

    def fixed_vertex_block(self, node: int) -> int:
        """
        Block to which the node is fixed (-1 if not fixed)
        """

    def get_partition(self) -> list[int]:
        """
        Returns a list with the block to which each node is assigned.
        """

    def imbalance(self, context: Context) -> float:
        """
        Computes the imbalance of the partition
        """

    def improve_mapping(
        self, target_graph: TargetGraph, context: Context, num_vcycles: int
    ) -> None:
        """
        Improves a mapping onto a graph using the iterated multilevel cycle technique (V-cycles)
        """

    def improve_partition(self, context: Context, num_vcycles: int) -> None:
        """
        Improves the partition using the iterated multilevel cycle technique (V-cycles)
        """

    def is_compatible(self, preset: PresetType) -> bool:
        """
        Returns whether or not improving the given partitioned hypergraph is compatible with the preset
        """

    def is_fixed(self, node: int) -> bool:
        """
        Returns whether or not the corresponding node is a fixed vertex
        """

    def is_incident_to_cut_edge(self, node: int) -> bool:
        """
        Returns true, if the corresponding node is incident to at least one cut hyperedge
        """

    def km1(self) -> int:
        """
        Computes the connectivity metric of the partition
        """

    def num_blocks(self) -> int:
        """
        Number of blocks of the partition
        """

    def num_incident_cut_edges(self, node: int) -> int:
        """
        Number of incident cut hyperedges of the corresponding node
        """

    def num_pins_in_block(self, hyperedge: int, block_id: int) -> int:
        """
        Number of pins assigned to the corresponding block in the given hyperedge
        """

    def soed(self) -> int:
        """
        Computes the sum-of-external-degree metric of the partition
        """

    def steiner_tree(self, target_graph: TargetGraph) -> int:
        """
        Computes the sum-of-external-degree metric of the partition
        """

    def write_partition_to_file(self, partition_file: str) -> None:
        """
        Writes the partition to a file
        """

class PresetType:
    """
    Members:

      DETERMINISTIC

      LARGE_K

      DEFAULT

      QUALITY

      HIGHEST_QUALITY
    """

    DEFAULT: typing.ClassVar[PresetType]  # value = <PresetType.DEFAULT: 2>
    DETERMINISTIC: typing.ClassVar[
        PresetType
    ]  # value = <PresetType.DETERMINISTIC: 0>
    HIGHEST_QUALITY: typing.ClassVar[
        PresetType
    ]  # value = <PresetType.HIGHEST_QUALITY: 4>
    LARGE_K: typing.ClassVar[PresetType]  # value = <PresetType.LARGE_K: 1>
    QUALITY: typing.ClassVar[PresetType]  # value = <PresetType.QUALITY: 3>
    __members__: typing.ClassVar[
        dict[str, PresetType]
    ]  # value = {'DETERMINISTIC': <PresetType.DETERMINISTIC: 0>, 'LARGE_K': <PresetType.LARGE_K: 1>, 'DEFAULT': <PresetType.DEFAULT: 2>, 'QUALITY': <PresetType.QUALITY: 3>, 'HIGHEST_QUALITY': <PresetType.HIGHEST_QUALITY: 4>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SystemError(Exception):
    pass

class TargetGraph(Graph):
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs): ...

class UnsupportedOperationError(Exception):
    pass

def initialize(*args, **kwargs) -> Initializer:
    """
    Initializes Mt-KaHyPar with the given number of threads.
    """

def set_seed(seed: int) -> None:
    """
    Initializes the random number generator with the given seed
    """

__version__: str = "dev"
