import torch.distributed as dist


def create_groups_with_cp_ranks(config, cpg_ranks):
    global_ranks = [
        [config.curr_cpg_ranks[r] for r in cpg_ranks_per_group]
        for cpg_ranks_per_group in cpg_ranks
    ]
    if dist.get_rank(group=config.cp_group) == 0:
        all_ranks_across_cp_groups = [None] * config.dp_degree
        dist.all_gather_object(
            all_ranks_across_cp_groups, global_ranks, group=config.dp_group
        )
        dist.broadcast_object_list(
            all_ranks_across_cp_groups,
            src=dist.get_global_rank(config.cp_group, 0),
            group=config.cp_group,
        )
    else:
        all_ranks_across_cp_groups = [None] * config.dp_degree
        dist.broadcast_object_list(
            all_ranks_across_cp_groups,
            src=dist.get_global_rank(config.cp_group, 0),
            group=config.cp_group,
        )
    global_groups = [
        [dist.new_group(ranks, backend="nccl") for ranks in cpg_ranks]
        for cpg_ranks in all_ranks_across_cp_groups
    ]
    return global_groups[config.dp_rank]
