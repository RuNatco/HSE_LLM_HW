import torch
import torch.distributed as dist


def _ring_neighbors() -> tuple[int, int]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return (rank - 1) % world_size, (rank + 1) % world_size


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.ndim != 1 or tensor.numel() % world_size:
        raise ValueError("Expected a 1D tensor with length divisible by world_size")
    prev_rank, next_rank = _ring_neighbors()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]

    if world_size == 1:
        return chunks[rank]

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        recv_buffer = torch.empty_like(chunks[recv_index])
        requests = dist.batch_isend_irecv([dist.P2POp(dist.isend, chunks[send_index], next_rank), dist.P2POp(dist.irecv, recv_buffer, prev_rank),])
        for request in requests:
            request.wait()
        chunks[recv_index] += recv_buffer

    return chunks[rank]


def ring_all_gather(shard):
    if shard.ndim != 1:
        raise ValueError("Expected a 1D shard")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    prev_rank, next_rank = _ring_neighbors()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        requests = dist.batch_isend_irecv([dist.P2POp(dist.isend, chunks[send_index], next_rank), dist.P2POp(dist.irecv, chunks[recv_index], prev_rank),])
        for request in requests:
            request.wait()

    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
