"""Small complete multiwave plans for decision-log comparator tests."""

from __future__ import annotations


CTA_SCALARS = (
    "inner_dim=2048", "outer_dim=1024", "vector_width=4",
    "regs_buffer_bits=65536", "tmp_dtype_bits=32", "warp_size=32",
    "sm_count=128", "max_threads_per_block=1024", "max_grid_y=65535",
    "smem_total_bits=140288", "smem_limit_bits=811008",
)


def plans(batch: int, read: int, rows: int) -> list[tuple[int, ...]]:
    maximum = 512 if batch == 1 else 256
    grid = 256 if rows == 4 else 128
    result = []
    for threads in range(maximum, 127, -1):
        if threads & (threads - 1):
            continue
        inner_batch = (512 + threads - 1) // threads
        wanted_x = (2048 // read + grid - 1) // grid
        x = min(1 << (wanted_x - 1).bit_length(),
                ((wanted_x + 7) // 8) * 8)
        while threads % x:
            x = min(x + 8, threads)
        y = threads // x
        warps = (threads // 32) * (grid // 128)
        available = 64 if warps == 32 else 128 if warps == 16 else 255
        required = 40 + 4 * inner_batch
        result.append((threads, 4, inner_batch, x, y, grid, 4, read,
                       warps, required, available, x, y, grid))
    return result


def rows_request(batch: int, read: int) -> dict:
    maximum = 512 if batch == 1 else 256
    state = [
        "inner_outer_multi_wave.grid_rows.v1",
        "inner_dim=2048", "outer_dim=1024", "vector_width=4",
        f"selected_min_batch={batch}",
        f"selected_outer_read_width={read}",
        "min_threads=128", f"max_threads={maximum}", "thread_cap=512",
        *CTA_SCALARS[3:],
    ]
    criteria = {}
    for rows in (4, 8, 16):
        group = plans(batch, read, rows)
        criteria[f"r{rows}"] = (
            f"Require at least {rows} outer rows per grid-Y block "
            f"across {len(group)} complete CTA plans")
        for plan in group:
            state.append("plan=" + ":".join(map(str, (rows, 640, 640, *plan))))
    return {"state": "\n".join(state),
            "questions": {"IOP3.grid_rows": {
                "type": "choice",
                "instructions": "Choose nvFuser scheduler decision IOP3.grid_rows",
                "criteria": criteria}}}


def cta_state(batch: int, read: int) -> str:
    return "\n".join((
        "inner_outer_multi_wave.cta.v2", *CTA_SCALARS,
        f"selected_outer_read_width={read}",
        *("plan=" + ":".join(map(str, plan))
          for plan in plans(batch, read, 8))))
