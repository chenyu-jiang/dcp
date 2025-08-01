def add_dcp_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title="dcp specific arguments")

    group.add_argument(
        "--disable-dcp", action="store_false", dest="enable_dcp"
    )
    # cost models
    group.add_argument("--dcp-intra-node-lat", type=float, default=0.0)
    group.add_argument("--dcp-intra-node-bw", type=float, default=4800)
    group.add_argument("--dcp-inter-node-lat", type=float, default=0.0)
    group.add_argument("--dcp-inter-node-bw", type=float, default=400)
    group.add_argument("--dcp-max-tflops", type=float, default=280)
    group.add_argument("--dcp-max-mem-bw", type=float, default=1500)
    # data loader and planner
    group.add_argument(
        "--dcp-prefetch-planner-num-workers", type=int, default=16
    )
    group.add_argument(
        "--dcp-prefetch-listener-num-workers", type=int, default=2
    )
    group.add_argument("--dcp-block-size", type=int, default=512)
    group.add_argument("--dcp-head-block-size", type=int, default=1)
    group.add_argument("--dcp-use-block-size-heuristic", action="store_true")
    group.add_argument("--dcp-mem-imbalance-epsilon", type=float, default=0.2)
    group.add_argument("--dcp-comp-imbalance-epsilon", type=float, default=0.2)
    group.add_argument(
        "--dcp-inter-node-comp-imbalance-factor", type=float, default=5.0
    )

    group.add_argument(
        "--dcp-global-batch-size-in-tokens", type=int, required=True
    )
    # dataset
    group.add_argument("--dcp-dataset-split", type=str, default="train")
    group.add_argument("--dcp-dataset-text-key", type=str, default="input")
    # executor
    group.add_argument("--dcp-use-cudagraph", action="store_true")
    # mask
    group.add_argument("--dcp-mask-type", type=str, default="causal")

    return parser
