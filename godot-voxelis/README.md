# godot-voxelis

Godot bindings for the [Voxelis](https://github.com/WildPixelGames/voxelis) voxel engine - a high-performance Sparse Voxel Octree DAG library written in pure Rust.

## Features

- **High Performance**: Leverages Voxelis's SVO DAG for efficient voxel storage with up to 99.999% compression ratio
- **Batch Operations**: Support for batched voxel modifications (up to 224x faster than single operations)
- **Chunk-based World**: Built-in `VoxelWorld` class for managing large voxel worlds with automatic chunking
- **Shape Primitives**: Helpers for drawing lines, boxes, and spheres
- **Mesh Generation**: Built-in `VoxelMeshBuilder` for converting voxel data to renderable Godot meshes

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

### VoxelMeshBuilder

Converts voxel data into renderable Godot meshes using culled face generation (only visible faces are generated).

```gdscript
var mesh_builder = VoxelMeshBuilder.create()

# Customize voxel size (default: 1.0)
mesh_builder.set_voxel_size(1.0)

# Enable vertex colors (default: true)
mesh_builder.set_use_vertex_colors(true)

# Customize the color palette for voxel types
mesh_builder.set_voxel_color(1, Color.GRAY)      # Stone
mesh_builder.set_voxel_color(2, Color.SADDLE_BROWN)  # Dirt
mesh_builder.set_voxel_color(3, Color.GREEN)     # Grass

# Build mesh from a VoxelTree
var mesh = mesh_builder.build_mesh(interner, tree)

# Create MeshInstance3D to display it
var mesh_instance = MeshInstance3D.new()
mesh_instance.mesh = mesh
add_child(mesh_instance)

# For VoxelWorld, build meshes per chunk
var chunk_mesh = mesh_builder.build_chunk_mesh(interner, world, Vector3i(0, 0, 0))
```

### VoxelChunkRenderer

Helper class for managing chunk mesh generation and dirty tracking.

```gdscript
var renderer = VoxelChunkRenderer.create()
renderer.set_world(world)
renderer.set_interner(interner)

# Mark chunks as dirty when voxels change
world.set_voxel(interner, Vector3i(0, 0, 0), 1)
renderer.mark_dirty_at(Vector3i(0, 0, 0))

# Or mark a specific chunk
renderer.mark_chunk_dirty(Vector3i(0, 0, 0))

# Build meshes for dirty chunks
for chunk_pos in renderer.get_dirty_chunks():
    var mesh = renderer.build_chunk(chunk_pos)
    # Add mesh to scene...

renderer.clear_dirty()
```

## Rendering Your Voxel World

To render a voxel world, you need to convert the voxel data into meshes that Godot can display. The `VoxelMeshBuilder` class handles this conversion using **culled face generation** - it only creates polygons for voxel faces that are visible (not adjacent to other solid voxels).

### Basic Rendering Example

```gdscript
extends Node3D

var interner: VoxelInterner
var tree: VoxelTree
var mesh_builder: VoxelMeshBuilder

func _ready():
    interner = VoxelInterner.new()
    tree = VoxelTree.create(5)  # 32³ voxels
    mesh_builder = VoxelMeshBuilder.create()
    
    # Create some voxels
    var batch = tree.create_batch()
    batch.fill_sphere(interner, Vector3i(16, 16, 16), 10, 1)
    tree.apply_batch(interner, batch)
    
    # Build and display the mesh
    var mesh = mesh_builder.build_mesh(interner, tree)
    var mesh_instance = MeshInstance3D.new()
    mesh_instance.mesh = mesh
    add_child(mesh_instance)
```

### Rendering a Voxel World with Chunks

```gdscript
extends Node3D

var interner: VoxelInterner
var world: VoxelWorld
var mesh_builder: VoxelMeshBuilder
var chunk_meshes: Dictionary = {}  # chunk_pos -> MeshInstance3D

func _ready():
    interner = VoxelInterner.with_memory_budget(512 * 1024 * 1024)
    world = VoxelWorld.create(5)
    mesh_builder = VoxelMeshBuilder.create()
    
    # Setup custom colors
    mesh_builder.set_voxel_color(1, Color(0.5, 0.5, 0.5))  # Stone
    mesh_builder.set_voxel_color(2, Color(0.4, 0.25, 0.13))  # Dirt  
    mesh_builder.set_voxel_color(3, Color(0.2, 0.6, 0.2))  # Grass
    
    generate_terrain()
    render_all_chunks()

func generate_terrain():
    for x in range(-32, 32):
        for z in range(-32, 32):
            var height = int(8 + sin(x * 0.1) * 4 + cos(z * 0.1) * 4)
            for y in range(height):
                var voxel_type = 1 if y < height - 1 else 3
                world.set_voxel(interner, Vector3i(x, y, z), voxel_type)

func render_all_chunks():
    var chunk_size = world.get_chunk_size()
    
    for chunk_pos in world.get_loaded_chunks():
        var mesh = mesh_builder.build_chunk_mesh(interner, world, chunk_pos)
        
        var mesh_instance = MeshInstance3D.new()
        mesh_instance.mesh = mesh
        mesh_instance.position = Vector3(
            chunk_pos.x * chunk_size,
            chunk_pos.y * chunk_size,
            chunk_pos.z * chunk_size
        )
        add_child(mesh_instance)
        chunk_meshes[chunk_pos] = mesh_instance

func update_chunk(chunk_pos: Vector3i):
    # Remove old mesh if exists
    if chunk_meshes.has(chunk_pos):
        chunk_meshes[chunk_pos].queue_free()
    
    # Build new mesh
    var mesh = mesh_builder.build_chunk_mesh(interner, world, chunk_pos)
    var chunk_size = world.get_chunk_size()
    
    var mesh_instance = MeshInstance3D.new()
    mesh_instance.mesh = mesh
    mesh_instance.position = Vector3(
        chunk_pos.x * chunk_size,
        chunk_pos.y * chunk_size,
        chunk_pos.z * chunk_size
    )
    add_child(mesh_instance)
    chunk_meshes[chunk_pos] = mesh_instance
```

### Using Materials

The mesh builder generates vertex colors by default. To use them, create a material with vertex colors enabled:

```gdscript
# Create material with vertex colors
var material = StandardMaterial3D.new()
material.vertex_color_use_as_albedo = true

# Apply to mesh instance
mesh_instance.material_override = material
```

Or if you've compiled with `codegen-full` feature:

```gdscript
var material = VoxelMeshBuilder.create_default_material()
mesh_instance.material_override = material
```

### Default Color Palette

The mesh builder comes with a built-in color palette:

| Voxel Type | Color | Description |
|------------|-------|-------------|
| 0 | Transparent | Empty/Air |
| 1 | Gray | Stone |
| 2 | Brown | Dirt |
| 3 | Green | Grass |
| 4 | Tan | Sand |
| 5 | Blue | Water |
| 6 | Brown | Wood |
| 7 | Dark Green | Leaves |
| 8 | White | Snow |
| 9 | Orange | Copper |
| 10 | Yellow | Gold |
| 11 | Silver | Iron |
| 12 | Dark Gray | Coal |
| 13 | Red | Redstone |
| 14 | Cyan | Diamond |
| 15 | Purple | Amethyst |

## Building

```bash
# Debug build
cargo build -p godot-voxelis

# Release build
cargo build -p godot-voxelis --release

# With full Godot API (enables create_default_material)
cargo build -p godot-voxelis --release --features codegen-full
```

## Installation in Godot

1. Build the library for your target platform
2. Copy the compiled library to your Godot project (e.g., `addons/voxelis/`)
3. Copy `voxelis.gdextension` to your Godot project (e.g., `addons/voxelis/`)
4. Adjust library paths in the `.gdextension` file if needed
5. Restart Godot

## Example: Complete Voxel Game Setup

```gdscript
extends Node3D

var interner: VoxelInterner
var world: VoxelWorld
var mesh_builder: VoxelMeshBuilder
var chunk_meshes: Dictionary = {}

func _ready():
    # Initialize voxel system
    interner = VoxelInterner.with_memory_budget(512 * 1024 * 1024)
    world = VoxelWorld.create(5)  # 32³ chunks
    mesh_builder = VoxelMeshBuilder.create()
    
    # Generate initial terrain
    generate_terrain()
    
    # Render all chunks
    for chunk_pos in world.get_loaded_chunks():
        render_chunk(chunk_pos)

func generate_terrain():
    for x in range(-64, 64):
        for z in range(-64, 64):
            var height = int(16 + sin(x * 0.1) * 8 + cos(z * 0.1) * 8)
            for y in range(height):
                var voxel_type = 1 if y < height - 1 else 3
                world.set_voxel(interner, Vector3i(x, y, z), voxel_type)

func render_chunk(chunk_pos: Vector3i):
    var mesh = mesh_builder.build_chunk_mesh(interner, world, chunk_pos)
    var chunk_size = world.get_chunk_size()
    
    var mesh_instance = MeshInstance3D.new()
    mesh_instance.mesh = mesh
    mesh_instance.position = Vector3(
        chunk_pos.x * chunk_size,
        chunk_pos.y * chunk_size,
        chunk_pos.z * chunk_size
    )
    
    # Apply material with vertex colors
    var material = StandardMaterial3D.new()
    material.vertex_color_use_as_albedo = true
    mesh_instance.material_override = material
    
    add_child(mesh_instance)
    chunk_meshes[chunk_pos] = mesh_instance

func modify_voxel(world_pos: Vector3i, voxel_type: int):
    # Modify the voxel
    world.set_voxel(interner, world_pos, voxel_type)
    
    # Re-render affected chunk
    var chunk_pos = world.world_to_chunk(world_pos)
    if chunk_meshes.has(chunk_pos):
        chunk_meshes[chunk_pos].queue_free()
    render_chunk(chunk_pos)
```

## Performance Tips

1. **Use batches for bulk operations**: When modifying many voxels, use `VoxelBatch` instead of individual `set_voxel` calls
2. **Share the interner**: Use a single `VoxelInterner` instance across all trees/worlds for maximum memory deduplication
3. **Use appropriate chunk sizes**: Depth 5 (32³) is a good balance for most use cases
4. **Fill operations are O(1)**: `tree.fill()` is nearly instant regardless of tree size due to octree compression
5. **Only re-mesh dirty chunks**: Track which chunks have been modified and only regenerate their meshes
6. **Use release builds**: Mesh generation is significantly faster in release mode

## License

MIT OR Apache-2.0 (same as Voxelis)
