# godot-voxelis

Godot bindings for the [Voxelis](https://github.com/WildPixelGames/voxelis) voxel engine - a high-performance Sparse Voxel Octree DAG library written in pure Rust.

## Features

- **High Performance**: Leverages Voxelis's SVO DAG for efficient voxel storage with up to 99.999% compression ratio
- **Batch Operations**: Support for batched voxel modifications (up to 224x faster than single operations)
- **Chunk-based World**: Built-in `VoxelWorld` class for managing large voxel worlds with automatic chunking
- **Shape Primitives**: Helpers for drawing lines, boxes, and spheres
- **Mesh Generation**: Built-in `VoxelMeshBuilder` for converting voxel data to renderable Godot meshes
- **Asset System**: Create, save, load, and place reusable voxel structures with `VoxelAsset`
- **Procedural Generation**: Built-in generators for trees, rocks, buildings, and more

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

### VoxelAsset

Reusable voxel structures that can be created, saved, loaded, and placed.

```gdscript
# Create an asset manually
var asset = VoxelAsset.create()
asset.set_name("My Structure")
asset.set_voxel(Vector3i(0, 0, 0), 1)  # Base
asset.set_voxel(Vector3i(0, 1, 0), 1)  # Middle
asset.set_voxel(Vector3i(0, 2, 0), 2)  # Top

# Shape primitives
asset.fill_box(Vector3i(0, 0, 0), Vector3i(5, 3, 5), 1)
asset.fill_sphere(Vector3i(2, 5, 2), 3, 2)
asset.fill_cylinder(Vector3i(0, 0, 0), 2, 5, 1)

# Transform the asset
asset.center()       # Center around origin
asset.ground()       # Move so minimum Y is 0
asset.rotate_y_90()  # Rotate 90° around Y axis
asset.mirror_x()     # Mirror along X axis

# Place in world
asset.place_in_world(interner, world, Vector3i(10, 0, 10))

# Or place in a tree
asset.place_in_tree(interner, tree, Vector3i(5, 5, 5))

# Or use with batches for better performance
asset.place_in_batch(interner, batch, Vector3i(20, 0, 20))
tree.apply_batch(interner, batch)

# Save/load assets
var data = asset.to_dictionary()
# Save data to file using Godot's FileAccess...

var loaded_asset = VoxelAsset.from_dictionary(data)
```

### VoxelAssetGenerator

Procedurally generate common voxel structures.

```gdscript
# Trees
var tree = VoxelAssetGenerator.generate_tree(5, 3)        # trunk_height, canopy_radius
var pine = VoxelAssetGenerator.generate_pine_tree(10)      # height

# Natural features
var rock = VoxelAssetGenerator.generate_rock(4)            # size
var bush = VoxelAssetGenerator.generate_bush(2)            # size
var cactus = VoxelAssetGenerator.generate_cactus(6)        # height

# Buildings
var house = VoxelAssetGenerator.generate_house(8, 6, 10)   # width, height, depth
var wall = VoxelAssetGenerator.generate_wall(20, 4, 1)     # length, height, thickness
var pillar = VoxelAssetGenerator.generate_pillar(8, 2)     # height, radius
var stairs = VoxelAssetGenerator.generate_stairs(3, 8)     # width, steps

# Decorations
var fence = VoxelAssetGenerator.generate_fence(15, 3)      # length, height
var flower_bed = VoxelAssetGenerator.generate_flower_bed(5, 5)  # width, depth
var path = VoxelAssetGenerator.generate_path(20, 3)        # length, width

# Place in world
tree.place_in_world(interner, world, Vector3i(10, 0, 10))
rock.place_in_world(interner, world, Vector3i(20, 0, 15))
house.place_in_world(interner, world, Vector3i(30, 0, 20))
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

## Creating Voxel Assets

### Manual Asset Creation

Create custom structures by setting voxels directly:

```gdscript
func create_lamp_post() -> VoxelAsset:
    var asset = VoxelAsset.create()
    asset.set_name("Lamp Post")
    
    # Pole
    for y in range(5):
        asset.set_voxel(Vector3i(0, y, 0), 11)  # Iron
    
    # Lamp head
    asset.set_voxel(Vector3i(-1, 5, 0), 10)  # Gold
    asset.set_voxel(Vector3i(1, 5, 0), 10)
    asset.set_voxel(Vector3i(0, 5, -1), 10)
    asset.set_voxel(Vector3i(0, 5, 1), 10)
    asset.set_voxel(Vector3i(0, 6, 0), 10)
    
    return asset
```

### Asset Library Pattern

Organize your assets in a library:

```gdscript
class_name VoxelAssetLibrary

var assets: Dictionary = {}

func _init():
    # Load or generate assets
    assets["tree_oak"] = VoxelAssetGenerator.generate_tree(6, 4)
    assets["tree_pine"] = VoxelAssetGenerator.generate_pine_tree(12)
    assets["rock_small"] = VoxelAssetGenerator.generate_rock(2)
    assets["rock_large"] = VoxelAssetGenerator.generate_rock(5)
    assets["house_small"] = VoxelAssetGenerator.generate_house(6, 5, 6)
    assets["fence"] = VoxelAssetGenerator.generate_fence(10, 3)

func get_asset(name: String) -> VoxelAsset:
    if assets.has(name):
        return assets[name].duplicate()  # Return a copy
    return null

func place(name: String, interner: VoxelInterner, world: VoxelWorld, pos: Vector3i):
    var asset = get_asset(name)
    if asset:
        asset.place_in_world(interner, world, pos)
```

### Saving and Loading Assets

```gdscript
func save_asset(asset: VoxelAsset, path: String):
    var data = asset.to_dictionary()
    var file = FileAccess.open(path, FileAccess.WRITE)
    file.store_var(data)
    file.close()

func load_asset(path: String) -> VoxelAsset:
    var file = FileAccess.open(path, FileAccess.READ)
    var data = file.get_var()
    file.close()
    return VoxelAsset.from_dictionary(data)
```

### Populating a World with Assets

```gdscript
func populate_world():
    var library = VoxelAssetLibrary.new()
    
    # Scatter trees randomly
    for i in range(50):
        var pos = Vector3i(
            randi_range(-100, 100),
            0,
            randi_range(-100, 100)
        )
        # Get ground height at this position...
        var tree_type = "tree_oak" if randf() > 0.3 else "tree_pine"
        library.place(tree_type, interner, world, pos)
    
    # Add some rocks
    for i in range(30):
        var pos = Vector3i(
            randi_range(-100, 100),
            0,
            randi_range(-100, 100)
        )
        var rock_type = "rock_small" if randf() > 0.2 else "rock_large"
        library.place(rock_type, interner, world, pos)
    
    # Place a house
    library.place("house_small", interner, world, Vector3i(0, 0, 0))
```

## Performance Tips

1. **Use batches for bulk operations**: When modifying many voxels, use `VoxelBatch` instead of individual `set_voxel` calls
2. **Share the interner**: Use a single `VoxelInterner` instance across all trees/worlds for maximum memory deduplication
3. **Use appropriate chunk sizes**: Depth 5 (32³) is a good balance for most use cases
4. **Fill operations are O(1)**: `tree.fill()` is nearly instant regardless of tree size due to octree compression
5. **Only re-mesh dirty chunks**: Track which chunks have been modified and only regenerate their meshes
6. **Use release builds**: Mesh generation is significantly faster in release mode
7. **Use batches for placing assets**: When placing multiple assets, use `place_in_batch()` and apply once

## License

MIT OR Apache-2.0 (same as Voxelis)
