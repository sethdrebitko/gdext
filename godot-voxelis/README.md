# godot-voxelis

Godot bindings for the [Voxelis](https://github.com/WildPixelGames/voxelis) voxel engine - a high-performance Sparse Voxel Octree DAG library written in pure Rust.

## Features

- **High Performance**: Leverages Voxelis's SVO DAG for efficient voxel storage with up to 99.999% compression ratio
- **Batch Operations**: Support for batched voxel modifications (up to 224x faster than single operations)
- **Chunk-based World**: Built-in `VoxelWorld` class for managing large voxel worlds with automatic chunking
- **Shape Primitives**: Helpers for drawing lines, boxes, and spheres
- **Mesh Generation**: Built-in `VoxelMesher` for rendering voxels with culled-face and greedy meshing optimizations
- **Asset Pipeline**: Import MagicaVoxel models, define voxel palettes, and export voxel data

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

### VoxelMesher

Mesh generator for rendering voxel data. Converts voxel trees into Godot meshes.

```gdscript
var mesher = VoxelMesher.create()

# Configure voxel colors (voxel_type -> Color)
mesher.set_voxel_color(1, Color(0.5, 0.4, 0.3))  # Brown (dirt)
mesher.set_voxel_color(2, Color(0.3, 0.7, 0.3))  # Green (grass)
mesher.set_voxel_color(3, Color(0.3, 0.5, 0.9))  # Blue (water)

# Optional: Set voxel size (default: 1.0)
mesher.set_voxel_size(1.0)

# Optional: Enable greedy meshing for better performance
mesher.set_greedy_meshing(true)

# Generate mesh from a VoxelTree
var mesh = mesher.generate_mesh(interner, tree)
$MeshInstance3D.mesh = mesh

# Or generate mesh for a specific chunk in a VoxelWorld
var chunk_mesh = mesher.generate_chunk_mesh(interner, world, Vector3i(0, 0, 0))
```

### VoxelPalette

Define voxel types with colors, textures, and properties.

```gdscript
# Create a custom palette
var palette = VoxelPalette.create()
palette.add_type(1, "Stone", Color(0.5, 0.5, 0.55))
palette.add_type(2, "Dirt", Color(0.6, 0.45, 0.3))
palette.add_type(3, "Grass", Color(0.35, 0.65, 0.25))

# Or use the default terrain palette
var palette = VoxelPalette.create_default()

# Access voxel type properties
var color = palette.get_color(1)
var name = palette.get_name(1)
var is_transparent = palette.is_transparent(5)  # Water

# Set special properties
palette.set_transparent(5, true)  # Make water transparent
palette.set_emissive(9, true, Color(1.0, 0.5, 0.2))  # Make lava glow

# Texture atlas support (16x16 tiles)
palette.set_atlas_size(16)
palette.add_type_with_atlas(1, "Stone", Color.WHITE, 0, 0)  # Tile at (0,0)
palette.add_type_with_atlas(2, "Dirt", Color.WHITE, 1, 0)   # Tile at (1,0)

# Render with palette
var mesh = mesher.generate_mesh_with_palette(interner, tree, palette)

# Save/load palette
var data = palette.to_dictionary()
var loaded_palette = VoxelPalette.from_dictionary(data)
```

### VoxelImporter

Import voxel models from external files.

```gdscript
var importer = VoxelImporter.new()

# Load a MagicaVoxel .vox file
var result = importer.load_vox("res://models/house.vox", interner)
if result.success:
    var tree = result.tree       # VoxelTree with the model
    var palette = result.palette # VoxelPalette with colors from the file
    var size = result.size       # Vector3i model dimensions
    
    # Render the imported model
    var mesh = mesher.generate_mesh_with_palette(interner, tree, palette)
else:
    print("Failed to load: ", result.error)

# Import raw voxel data (x, y, z, type per voxel)
var raw_data = PackedByteArray([0, 0, 0, 1, 1, 0, 0, 2])  # Two voxels
var tree = importer.from_raw_data(interner, raw_data, 5)

# Export to raw data
var exported = importer.to_raw_data(interner, tree)
```

### VoxelExporter

Export voxel data to files.

```gdscript
var exporter = VoxelExporter.new()

# Export to MagicaVoxel .vox format
var success = exporter.save_vox("user://exported.vox", interner, tree, palette)
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

## Example: Terrain Generation with Rendering

```gdscript
extends Node3D

var interner: VoxelInterner
var world: VoxelWorld
var mesher: VoxelMesher
var chunk_meshes: Dictionary = {}

func _ready():
    interner = VoxelInterner.with_memory_budget(512 * 1024 * 1024)
    world = VoxelWorld.create(5)  # 32³ chunks
    
    # Setup mesher with terrain colors
    mesher = VoxelMesher.create()
    mesher.set_voxel_color(1, Color(0.4, 0.35, 0.3))  # Stone
    mesher.set_voxel_color(2, Color(0.3, 0.6, 0.2))   # Grass
    mesher.set_voxel_color(3, Color(0.6, 0.5, 0.3))   # Dirt
    mesher.set_greedy_meshing(true)  # Better performance
    
    generate_terrain()
    render_all_chunks()

func generate_terrain():
    # Generate a simple height-based terrain
    for x in range(-64, 64):
        for z in range(-64, 64):
            var height = int(16 + sin(x * 0.1) * 8 + cos(z * 0.1) * 8)
            for y in range(height):
                var voxel_type: int
                if y < height - 3:
                    voxel_type = 1  # Stone
                elif y < height - 1:
                    voxel_type = 3  # Dirt
                else:
                    voxel_type = 2  # Grass
                world.set_voxel(interner, Vector3i(x, y, z), voxel_type)
    
    print("Generated terrain with %d chunks" % world.get_chunk_count())

func render_all_chunks():
    # Create a material for the voxel meshes
    var material = StandardMaterial3D.new()
    material.vertex_color_use_as_albedo = true
    
    # Render each chunk
    for chunk_pos in world.get_loaded_chunks():
        var mesh = mesher.generate_chunk_mesh(interner, world, chunk_pos)
        if mesh.get_surface_count() > 0:
            var mesh_instance = MeshInstance3D.new()
            mesh_instance.mesh = mesh
            mesh_instance.material_override = material
            add_child(mesh_instance)
            chunk_meshes[chunk_pos] = mesh_instance
    
    print("Rendered %d chunk meshes" % chunk_meshes.size())

func update_chunk(chunk_pos: Vector3i):
    # Re-render a specific chunk after modification
    if chunk_meshes.has(chunk_pos):
        chunk_meshes[chunk_pos].queue_free()
    
    var mesh = mesher.generate_chunk_mesh(interner, world, chunk_pos)
    if mesh.get_surface_count() > 0:
        var mesh_instance = MeshInstance3D.new()
        mesh_instance.mesh = mesh
        var material = StandardMaterial3D.new()
        material.vertex_color_use_as_albedo = true
        mesh_instance.material_override = material
        add_child(mesh_instance)
        chunk_meshes[chunk_pos] = mesh_instance
```

## Example: Simple Single-Tree Rendering

```gdscript
extends Node3D

func _ready():
    var interner = VoxelInterner.new()
    var tree = VoxelTree.create(5)  # 32³ voxels
    
    # Create some voxels
    var batch = tree.create_batch()
    batch.fill_sphere(interner, Vector3i(16, 16, 16), 10, 1)
    tree.apply_batch(interner, batch)
    
    # Render the tree
    var mesher = VoxelMesher.create()
    mesher.set_voxel_color(1, Color.CORNFLOWER_BLUE)
    
    var mesh = mesher.generate_mesh(interner, tree)
    
    var mesh_instance = MeshInstance3D.new()
    mesh_instance.mesh = mesh
    
    var material = StandardMaterial3D.new()
    material.vertex_color_use_as_albedo = true
    mesh_instance.material_override = material
    
    add_child(mesh_instance)
```

## Performance Tips

### Voxel Operations
1. **Use batches for bulk operations**: When modifying many voxels, use `VoxelBatch` instead of individual `set_voxel` calls
2. **Share the interner**: Use a single `VoxelInterner` instance across all trees/worlds for maximum memory deduplication
3. **Use appropriate chunk sizes**: Depth 5 (32³) is a good balance for most use cases
4. **Fill operations are O(1)**: `tree.fill()` is nearly instant regardless of tree size due to octree compression

### Rendering
1. **Enable greedy meshing**: `mesher.set_greedy_meshing(true)` significantly reduces polygon count by merging adjacent faces with the same voxel type
2. **Only re-mesh modified chunks**: Track which chunks changed and only regenerate their meshes
3. **Use vertex colors**: Set `material.vertex_color_use_as_albedo = true` to use the built-in color system
4. **Consider LOD**: For distant chunks, use smaller depth trees or skip rendering entirely
5. **Background meshing**: For large worlds, consider generating meshes in a background thread

### Mesh Generation Performance
- **Culled-face meshing** (default): Only generates faces between solid and empty voxels. Good balance of speed and polygon count.
- **Greedy meshing**: Combines adjacent faces into larger quads. Slower to generate but produces significantly fewer polygons. Best for static terrain.

## Creating Assets

### Using MagicaVoxel (Recommended)

[MagicaVoxel](https://ephtracy.github.io/) is a free voxel editor that's perfect for creating voxel models.

1. **Download MagicaVoxel** from https://ephtracy.github.io/
2. **Create your model** using the editor
3. **Save as .vox file** (File > Save)
4. **Import in Godot**:

```gdscript
var importer = VoxelImporter.new()
var result = importer.load_vox("res://models/my_model.vox", interner)
if result.success:
    var mesh = mesher.generate_mesh_with_palette(interner, result.tree, result.palette)
    $MeshInstance3D.mesh = mesh
```

### Creating a Texture Atlas

For textured voxels, create a texture atlas:

1. **Create a tilemap image** (e.g., 256x256 for 16x16 tiles at 16px each)
2. **Arrange textures in a grid**:
   - Tile (0,0) = Stone texture
   - Tile (1,0) = Dirt texture
   - Tile (2,0) = Grass texture
   - etc.

3. **Configure the palette**:

```gdscript
var palette = VoxelPalette.create()
palette.set_atlas_size(16)  # 16x16 tiles
palette.add_type_with_atlas(1, "Stone", Color.WHITE, 0, 0)
palette.add_type_with_atlas(2, "Dirt", Color.WHITE, 1, 0)
palette.add_type_with_atlas(3, "Grass", Color.WHITE, 2, 0)
```

4. **Apply the texture to your material**:

```gdscript
var material = StandardMaterial3D.new()
material.albedo_texture = preload("res://textures/voxel_atlas.png")
material.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST  # Crisp pixels
material.vertex_color_use_as_albedo = true  # Tint with palette colors
mesh_instance.material_override = material
```

### Procedural Generation

Generate voxels programmatically:

```gdscript
func generate_tree(interner, world, base_pos: Vector3i):
    # Trunk
    for y in range(5):
        world.set_voxel(interner, base_pos + Vector3i(0, y, 0), 6)  # Wood
    
    # Leaves (sphere)
    var batch = world.get_or_create_chunk(world.world_to_chunk(base_pos)).bind().create_batch()
    var leaf_center = base_pos + Vector3i(0, 6, 0)
    for x in range(-2, 3):
        for y in range(-2, 3):
            for z in range(-2, 3):
                if x*x + y*y + z*z <= 6:
                    world.set_voxel(interner, leaf_center + Vector3i(x, y, z), 7)  # Leaves

func generate_terrain_with_noise(interner, world):
    var noise = FastNoiseLite.new()
    noise.seed = randi()
    noise.frequency = 0.02
    
    for x in range(-128, 128):
        for z in range(-128, 128):
            var height = int(32 + noise.get_noise_2d(x, z) * 16)
            for y in range(height):
                var voxel_type: int
                if y < height - 4:
                    voxel_type = 1  # Stone
                elif y < height - 1:
                    voxel_type = 2  # Dirt
                else:
                    voxel_type = 3  # Grass
                world.set_voxel(interner, Vector3i(x, y, z), voxel_type)
```

### Default Voxel Types

`VoxelPalette.create_default()` includes these types:

| ID | Name   | Color         | Properties |
|----|--------|---------------|------------|
| 1  | Stone  | Gray          | Solid      |
| 2  | Dirt   | Brown         | Solid      |
| 3  | Grass  | Green         | Solid      |
| 4  | Sand   | Yellow        | Solid      |
| 5  | Water  | Blue          | Transparent|
| 6  | Wood   | Brown         | Solid      |
| 7  | Leaves | Green         | Transparent|
| 8  | Ore    | Gold          | Solid      |
| 9  | Lava   | Orange        | Emissive   |
| 10 | Glass  | Light Blue    | Transparent|

### Saving and Loading Voxel Data

```gdscript
# Save palette to JSON
func save_palette(palette: VoxelPalette, path: String):
    var file = FileAccess.open(path, FileAccess.WRITE)
    file.store_string(JSON.stringify(palette.to_dictionary()))

# Load palette from JSON
func load_palette(path: String) -> VoxelPalette:
    var file = FileAccess.open(path, FileAccess.READ)
    var data = JSON.parse_string(file.get_as_text())
    return VoxelPalette.from_dictionary(data)

# Export voxel model
func save_model(interner, tree, palette, path: String):
    var exporter = VoxelExporter.new()
    exporter.save_vox(path, interner, tree, palette)
```

## License

MIT OR Apache-2.0 (same as Voxelis)
