# godot-voxelis

Godot bindings for the [Voxelis](https://github.com/WildPixelGames/voxelis) voxel engine - a high-performance Sparse Voxel Octree DAG library written in pure Rust.

## Features

- **High Performance**: Leverages Voxelis's SVO DAG for efficient voxel storage with up to 99.999% compression ratio
- **Batch Operations**: Support for batched voxel modifications (up to 224x faster than single operations)
- **Chunk-based World**: Built-in `VoxelWorld` class for managing large voxel worlds with automatic chunking
- **Shape Primitives**: Helpers for drawing lines, boxes, and spheres

## Classes

### VoxelInterner

Shared memory manager for voxel data using hash-consing compression.

```gdscript
var interner = VoxelInterner.new()
# Or with custom memory budget (default: 256 MB)
var interner = VoxelInterner.with_memory_budget(512 * 1024 * 1024)
```

### VoxelTree

A high-performance Sparse Voxel Octree DAG for voxel storage.

```gdscript
var interner = VoxelInterner.new()
var tree = VoxelTree.create(5)  # 32³ voxels (2^5 = 32)

# Single voxel operations
tree.set_voxel(interner, Vector3i(3, 0, 4), 1)
var value = tree.get_voxel(interner, Vector3i(3, 0, 4))

# Fill entire tree (O(1) operation!)
tree.fill(interner, 1)

# Clear the tree
tree.clear(interner)
```

### VoxelBatch

Batch container for efficient bulk voxel operations.

```gdscript
var batch = tree.create_batch()

# Individual set operations
batch.set_voxel(interner, Vector3i(0, 0, 0), 1)
batch.set_voxel(interner, Vector3i(1, 1, 1), 2)

# Shape primitives
batch.fill_box(interner, Vector3i(0, 0, 0), Vector3i(10, 10, 10), 1)
batch.fill_sphere(interner, Vector3i(16, 16, 16), 8, 2)
batch.set_line(interner, Vector3i(0, 0, 0), Vector3i(31, 31, 31), 3)

# Apply all operations at once (much faster!)
tree.apply_batch(interner, batch)
```

### VoxelWorld

Chunk-based voxel world manager for large worlds.

```gdscript
var interner = VoxelInterner.new()
var world = VoxelWorld.create(5)  # Chunks of 32³ voxels

# World coordinates (automatically maps to chunks)
world.set_voxel(interner, Vector3i(100, 50, -200), 1)
var value = world.get_voxel(interner, Vector3i(100, 50, -200))

# Chunk management
var chunk_count = world.get_chunk_count()
var loaded_chunks = world.get_loaded_chunks()
world.unload_chunk(Vector3i(0, 0, 0))
```

## Building

```bash
# Debug build
cargo build -p godot-voxelis

# Release build
cargo build -p godot-voxelis --release
```

## Installation in Godot

1. Build the library for your target platform
2. Copy the compiled library to your Godot project (e.g., `addons/voxelis/`)
3. Copy `voxelis.gdextension` to your Godot project (e.g., `addons/voxelis/`)
4. Adjust library paths in the `.gdextension` file if needed
5. Restart Godot

## Example: Terrain Generation

```gdscript
extends Node3D

var interner: VoxelInterner
var world: VoxelWorld

func _ready():
    interner = VoxelInterner.with_memory_budget(512 * 1024 * 1024)
    world = VoxelWorld.create(5)  # 32³ chunks
    
    generate_terrain()

func generate_terrain():
    # Generate a simple height-based terrain
    for x in range(-64, 64):
        for z in range(-64, 64):
            var height = int(16 + sin(x * 0.1) * 8 + cos(z * 0.1) * 8)
            for y in range(height):
                var voxel_type = 1 if y < height - 1 else 2  # Stone below, grass on top
                world.set_voxel(interner, Vector3i(x, y, z), voxel_type)
    
    print("Generated terrain with %d chunks" % world.get_chunk_count())
```

## Performance Tips

1. **Use batches for bulk operations**: When modifying many voxels, use `VoxelBatch` instead of individual `set_voxel` calls
2. **Share the interner**: Use a single `VoxelInterner` instance across all trees/worlds for maximum memory deduplication
3. **Use appropriate chunk sizes**: Depth 5 (32³) is a good balance for most use cases
4. **Fill operations are O(1)**: `tree.fill()` is nearly instant regardless of tree size due to octree compression

## License

MIT OR Apache-2.0 (same as Voxelis)
