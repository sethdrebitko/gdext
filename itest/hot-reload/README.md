## Hot-reload Godot project

This directory hosts the Godot project used for gdext hot-reload tests. It also
includes an optional Voxelis integration for voxel storage.

### Voxelis integration

Enable the feature on the Rust crate to build with Voxelis:

```bash
cargo build -p hot-reload --features voxelis
```

This registers a `VoxelWorld` class with the following API:

- `configure(max_depth, memory_budget_mb)` to reinitialize storage.
- `set_voxel(position: Vector3i, value: int)` returns false on invalid input.
- `get_voxel(position: Vector3i)` returns -1 if out of bounds, 0 for empty.
- `fill(value: int)` and `clear()` for bulk operations.
- `get_voxel_bounds()` to query the valid axis size (1 << max_depth).

Note: Voxelis currently requires Rust 1.88+ and glam 0.29.
