from dcp.core.steiner_tree_packing import pack_multicast_trees


res = pack_multicast_trees(
    4,
    [(0, [1, 2, 3]), (0, [1, 2, 3]), (0, [1, 2, 3]), (0, [1, 2, 3])],
    [1, 1, 1, 1],
    3,
)
