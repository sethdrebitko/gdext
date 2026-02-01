//! Godot bindings for the Voxelis voxel engine.
//!
//! This crate provides Godot-compatible wrappers for the Voxelis Sparse Voxel Octree DAG engine,
//! enabling efficient voxel manipulation in Godot games.
//!
//! # Example usage in GDScript:
//! ```gdscript
//! var interner = VoxelInterner.new()
//! interner.set_memory_budget(256 * 1024 * 1024)  # 256 MB
//!
//! var tree = VoxelTree.new()
//! tree.initialize(5)  # 32³ voxels (2^5 = 32)
//!
//! # Single voxel operations
//! tree.set_voxel(interner, Vector3i(3, 0, 4), 1)
//! var value = tree.get_voxel(interner, Vector3i(3, 0, 4))
//!
//! # Batch operations (much faster for multiple edits)
//! var batch = tree.create_batch()
//! batch.set_voxel(interner, Vector3i(0, 0, 0), 1)
//! batch.set_voxel(interner, Vector3i(1, 1, 1), 2)
//! tree.apply_batch(interner, batch)
//!
//! # Rendering - Generate meshes from voxel data
//! var mesh_builder = VoxelMeshBuilder.new()
//! var mesh = mesh_builder.build_mesh(interner, tree)
//! var mesh_instance = MeshInstance3D.new()
//! mesh_instance.mesh = mesh
//! add_child(mesh_instance)
//! ```

use std::sync::{Arc, RwLock};

use glam::IVec3;
use godot::builtin::varray;
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::ArrayMesh;
use godot::obj::{EngineEnum, NewGd};
use godot::prelude::*;

use voxelis::spatial::{VoxOpsBatch, VoxOpsBulkWrite, VoxOpsRead, VoxOpsWrite, VoxTree};
use voxelis::{Batch, MaxDepth, VoxInterner};

/// Extension entry point for the Voxelis GDExtension.
struct VoxelisExtension;

#[gdextension]
unsafe impl ExtensionLibrary for VoxelisExtension {}

// ============================================================================
// VoxelInterner - Shared memory manager for voxel data
// ============================================================================

/// A shared memory interner for voxel data using hash-consing compression.
///
/// The VoxelInterner manages memory for all voxel trees, providing efficient
/// deduplication of identical voxel patterns. This enables massive memory savings
/// (up to 99.999% compression ratio) for worlds with repetitive patterns.
///
/// # Memory Budget
/// Set the memory budget before creating trees to control maximum memory usage.
/// Default is 256 MB.
#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct VoxelInterner {
    /// Internal voxelis interner wrapped in Arc<RwLock> for thread-safe sharing
    interner: Arc<RwLock<VoxInterner<u16>>>,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for VoxelInterner {
    fn init(base: Base<RefCounted>) -> Self {
        // Default memory budget: 256 MB
        let interner = VoxInterner::<u16>::with_memory_budget(256 * 1024 * 1024);
        Self {
            interner: Arc::new(RwLock::new(interner)),
            base,
        }
    }
}

#[godot_api]
impl VoxelInterner {
    /// Creates a new VoxelInterner with the specified memory budget in bytes.
    ///
    /// # Arguments
    /// * `memory_budget` - Maximum memory usage in bytes (default: 256 MB)
    #[func]
    pub fn with_memory_budget(memory_budget: i64) -> Gd<Self> {
        let interner = VoxInterner::<u16>::with_memory_budget(memory_budget as usize);
        Gd::from_init_fn(|base| Self {
            interner: Arc::new(RwLock::new(interner)),
            base,
        })
    }

    /// Sets a new memory budget for the interner.
    ///
    /// Note: This creates a new interner - existing trees should be recreated.
    #[func]
    pub fn set_memory_budget(&mut self, memory_budget: i64) {
        let new_interner = VoxInterner::<u16>::with_memory_budget(memory_budget as usize);
        *self.interner.write().unwrap() = new_interner;
    }

    /// Returns statistics about memory usage.
    #[func]
    pub fn get_stats(&self) -> VarDictionary {
        let _interner = self.interner.read().unwrap();
        let mut dict = VarDictionary::new();
        // Note: Stats may vary based on voxelis version
        dict.set("type", "VoxInterner<u16>");
        dict
    }

    /// Gets a clone of the internal Arc for sharing between trees
    pub(crate) fn get_arc(&self) -> Arc<RwLock<VoxInterner<u16>>> {
        Arc::clone(&self.interner)
    }
}

// ============================================================================
// VoxelTree - High-performance Sparse Voxel Octree DAG
// ============================================================================

/// A high-performance Sparse Voxel Octree DAG (Directed Acyclic Graph) for voxel storage.
///
/// VoxelTree provides efficient storage and manipulation of 3D voxel data using
/// a compressed octree structure with hash-consing for deduplication.
///
/// # Depth and Resolution
/// The tree depth determines the resolution:
/// - Depth 5: 32³ voxels (typical chunk size)
/// - Depth 6: 64³ voxels
/// - Depth 7: 128³ voxels
/// - Depth 8: 256³ voxels
///
/// # Usage
/// 1. Create a VoxelInterner to manage memory
/// 2. Create a VoxelTree with desired depth
/// 3. Use set_voxel/get_voxel for single operations
/// 4. Use batches for bulk operations (much faster)
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelTree {
    /// Internal voxelis tree
    tree: Option<VoxTree<u16>>,
    /// Maximum depth of the tree
    max_depth: u8,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelTree {
    /// Initializes the voxel tree with the specified depth.
    ///
    /// # Arguments
    /// * `depth` - Tree depth (1-7). Resolution is 2^depth per axis.
    ///   - 5 = 32³ voxels
    ///   - 6 = 64³ voxels
    ///   - 7 = 128³ voxels
    #[func]
    pub fn initialize(&mut self, depth: i32) {
        let depth = depth.clamp(1, 7) as u8;
        self.max_depth = depth;
        self.tree = Some(VoxTree::new(MaxDepth::new(depth)));
    }

    /// Creates a new VoxelTree with the specified depth.
    ///
    /// # Arguments
    /// * `depth` - Tree depth (1-7). Resolution is 2^depth per axis.
    #[func]
    pub fn create(depth: i32) -> Gd<Self> {
        let depth = depth.clamp(1, 7) as u8;
        let tree = VoxTree::new(MaxDepth::new(depth));
        Gd::from_init_fn(|base| Self {
            tree: Some(tree),
            max_depth: depth,
            base,
        })
    }

    /// Returns the maximum depth of this tree.
    #[func]
    pub fn get_depth(&self) -> i32 {
        self.max_depth as i32
    }

    /// Returns the resolution (size per axis) of this tree.
    #[func]
    pub fn get_resolution(&self) -> i32 {
        1 << self.max_depth
    }

    /// Gets the voxel value at the specified position.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `position` - 3D position within the tree bounds
    ///
    /// # Returns
    /// The voxel value at the position, or -1 if empty/out of bounds.
    #[func]
    pub fn get_voxel(&self, interner: Gd<VoxelInterner>, position: Vector3i) -> i32 {
        let Some(ref tree) = self.tree else {
            godot_error!("VoxelTree not initialized. Call initialize() first.");
            return -1;
        };

        let pos = IVec3::new(position.x, position.y, position.z);
        let max = 1 << self.max_depth;

        if pos.x < 0 || pos.x >= max || pos.y < 0 || pos.y >= max || pos.z < 0 || pos.z >= max {
            return -1;
        }

        let interner_arc = interner.bind().get_arc();
        let interner_guard = interner_arc.read().unwrap();

        match tree.get(&interner_guard, pos) {
            Some(value) => value as i32,
            None => -1,
        }
    }

    /// Sets the voxel value at the specified position.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `position` - 3D position within the tree bounds
    /// * `value` - The voxel value to set (0 = empty, 1-65535 = voxel types)
    #[func]
    pub fn set_voxel(&mut self, interner: Gd<VoxelInterner>, position: Vector3i, value: i32) {
        let Some(ref mut tree) = self.tree else {
            godot_error!("VoxelTree not initialized. Call initialize() first.");
            return;
        };

        let pos = IVec3::new(position.x, position.y, position.z);
        let max = 1 << self.max_depth;

        if pos.x < 0 || pos.x >= max || pos.y < 0 || pos.y >= max || pos.z < 0 || pos.z >= max {
            godot_warn!(
                "VoxelTree::set_voxel: Position {:?} out of bounds (max: {})",
                pos,
                max
            );
            return;
        }

        let interner_arc = interner.bind().get_arc();
        let mut interner_guard = interner_arc.write().unwrap();

        tree.set(&mut interner_guard, pos, value.clamp(0, 65535) as u16);
    }

    /// Fills the entire tree with a single value.
    ///
    /// This is extremely fast due to octree compression - O(1) operation.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `value` - The voxel value to fill with
    #[func]
    pub fn fill(&mut self, interner: Gd<VoxelInterner>, value: i32) {
        let Some(ref mut tree) = self.tree else {
            godot_error!("VoxelTree not initialized. Call initialize() first.");
            return;
        };

        let interner_arc = interner.bind().get_arc();
        let mut interner_guard = interner_arc.write().unwrap();

        tree.fill(&mut interner_guard, value.clamp(0, 65535) as u16);
    }

    /// Clears the entire tree (fills with 0/empty).
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    #[func]
    pub fn clear(&mut self, interner: Gd<VoxelInterner>) {
        self.fill(interner, 0);
    }

    /// Creates a batch for efficient bulk voxel operations.
    ///
    /// Batch operations are significantly faster than individual set_voxel calls
    /// when modifying many voxels (up to 224x faster).
    ///
    /// # Returns
    /// A new VoxelBatch that can be filled with operations and applied.
    #[func]
    pub fn create_batch(&self) -> Gd<VoxelBatch> {
        let Some(ref tree) = self.tree else {
            godot_error!("VoxelTree not initialized. Call initialize() first.");
            return Gd::from_init_fn(|base| VoxelBatch {
                batch: None,
                max_depth: 5,
                base,
            });
        };

        let batch = tree.create_batch();
        Gd::from_init_fn(|base| VoxelBatch {
            batch: Some(batch),
            max_depth: self.max_depth,
            base,
        })
    }

    /// Applies a batch of voxel operations to the tree.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `batch` - The VoxelBatch containing operations to apply
    #[func]
    pub fn apply_batch(&mut self, interner: Gd<VoxelInterner>, batch: Gd<VoxelBatch>) {
        let Some(ref mut tree) = self.tree else {
            godot_error!("VoxelTree not initialized. Call initialize() first.");
            return;
        };

        let batch_bind = batch.bind();
        let Some(ref batch_inner) = batch_bind.batch else {
            godot_error!("VoxelBatch not initialized.");
            return;
        };

        let interner_arc = interner.bind().get_arc();
        let mut interner_guard = interner_arc.write().unwrap();

        tree.apply_batch(&mut interner_guard, batch_inner);
    }

    /// Checks if the tree contains any non-empty voxels.
    #[func]
    pub fn is_empty(&self) -> bool {
        match &self.tree {
            Some(tree) => tree.get_root_id().is_empty(),
            None => true,
        }
    }
}

// ============================================================================
// VoxelBatch - Batch operations for efficient bulk voxel manipulation
// ============================================================================

/// A batch container for efficient bulk voxel operations.
///
/// Using batches is significantly faster than individual set_voxel calls
/// when modifying many voxels. Benchmarks show up to 224x speedup for
/// uniform patterns.
///
/// # Usage
/// ```gdscript
/// var batch = tree.create_batch()
/// for x in range(32):
///     for y in range(32):
///         for z in range(32):
///             batch.set_voxel(interner, Vector3i(x, y, z), 1)
/// tree.apply_batch(interner, batch)
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelBatch {
    /// Internal voxelis batch
    batch: Option<Batch<u16>>,
    /// Maximum depth inherited from parent tree
    max_depth: u8,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelBatch {
    /// Adds a voxel set operation to the batch.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `position` - 3D position within tree bounds
    /// * `value` - The voxel value to set
    #[func]
    pub fn set_voxel(&mut self, interner: Gd<VoxelInterner>, position: Vector3i, value: i32) {
        let Some(ref mut batch) = self.batch else {
            godot_error!("VoxelBatch not initialized. Create via VoxelTree.create_batch()");
            return;
        };

        let pos = IVec3::new(position.x, position.y, position.z);
        let max = 1 << self.max_depth;

        if pos.x < 0 || pos.x >= max || pos.y < 0 || pos.y >= max || pos.z < 0 || pos.z >= max {
            godot_warn!(
                "VoxelBatch::set_voxel: Position {:?} out of bounds (max: {})",
                pos,
                max
            );
            return;
        }

        let interner_arc = interner.bind().get_arc();
        let mut interner_guard = interner_arc.write().unwrap();

        batch.set(&mut interner_guard, pos, value.clamp(0, 65535) as u16);
    }

    /// Adds multiple voxel set operations along a line.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `start` - Starting position
    /// * `end` - Ending position
    /// * `value` - The voxel value to set
    #[func]
    pub fn set_line(
        &mut self,
        interner: Gd<VoxelInterner>,
        start: Vector3i,
        end: Vector3i,
        value: i32,
    ) {
        // Simple line drawing using Bresenham-like approach
        let dx = (end.x - start.x).abs();
        let dy = (end.y - start.y).abs();
        let dz = (end.z - start.z).abs();

        let sx = if start.x < end.x { 1 } else { -1 };
        let sy = if start.y < end.y { 1 } else { -1 };
        let sz = if start.z < end.z { 1 } else { -1 };

        let dm = dx.max(dy).max(dz);
        let mut x = start.x;
        let mut y = start.y;
        let mut z = start.z;

        for _ in 0..=dm {
            self.set_voxel(interner.clone(), Vector3i::new(x, y, z), value);

            if dm == dx {
                x += sx;
                if 2 * (y - start.y) * dx > dy * (x - start.x) * 2 - dy {
                    y += sy;
                }
                if 2 * (z - start.z) * dx > dz * (x - start.x) * 2 - dz {
                    z += sz;
                }
            } else if dm == dy {
                y += sy;
                if 2 * (x - start.x) * dy > dx * (y - start.y) * 2 - dx {
                    x += sx;
                }
                if 2 * (z - start.z) * dy > dz * (y - start.y) * 2 - dz {
                    z += sz;
                }
            } else {
                z += sz;
                if 2 * (x - start.x) * dz > dx * (z - start.z) * 2 - dx {
                    x += sx;
                }
                if 2 * (y - start.y) * dz > dy * (z - start.z) * 2 - dy {
                    y += sy;
                }
            }
        }
    }

    /// Fills a box region with a voxel value.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `min_corner` - Minimum corner of the box
    /// * `max_corner` - Maximum corner of the box (inclusive)
    /// * `value` - The voxel value to set
    #[func]
    pub fn fill_box(
        &mut self,
        interner: Gd<VoxelInterner>,
        min_corner: Vector3i,
        max_corner: Vector3i,
        value: i32,
    ) {
        let min_x = min_corner.x.min(max_corner.x);
        let max_x = min_corner.x.max(max_corner.x);
        let min_y = min_corner.y.min(max_corner.y);
        let max_y = min_corner.y.max(max_corner.y);
        let min_z = min_corner.z.min(max_corner.z);
        let max_z = min_corner.z.max(max_corner.z);

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    self.set_voxel(interner.clone(), Vector3i::new(x, y, z), value);
                }
            }
        }
    }

    /// Fills a sphere region with a voxel value.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `center` - Center of the sphere
    /// * `radius` - Radius of the sphere
    /// * `value` - The voxel value to set
    #[func]
    pub fn fill_sphere(
        &mut self,
        interner: Gd<VoxelInterner>,
        center: Vector3i,
        radius: i32,
        value: i32,
    ) {
        let radius_sq = radius * radius;

        for x in (center.x - radius)..=(center.x + radius) {
            for y in (center.y - radius)..=(center.y + radius) {
                for z in (center.z - radius)..=(center.z + radius) {
                    let dx = x - center.x;
                    let dy = y - center.y;
                    let dz = z - center.z;
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq <= radius_sq {
                        self.set_voxel(interner.clone(), Vector3i::new(x, y, z), value);
                    }
                }
            }
        }
    }

    /// Returns whether this batch has any pending operations.
    #[func]
    pub fn has_operations(&self) -> bool {
        match &self.batch {
            Some(batch) => {
                // Check if any mask has non-zero values
                batch.masks().iter().any(|(set, clear)| *set != 0 || *clear != 0)
            }
            None => false,
        }
    }

    /// Clears all operations from the batch.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    #[func]
    pub fn clear(&mut self, interner: Gd<VoxelInterner>) {
        if let Some(ref mut batch) = self.batch {
            let interner_arc = interner.bind().get_arc();
            let mut interner_guard = interner_arc.write().unwrap();
            batch.clear(&mut interner_guard);
        }
    }
}

// ============================================================================
// VoxelWorld - Chunk-based world management
// ============================================================================

/// A chunk-based voxel world manager.
///
/// VoxelWorld manages multiple VoxelTrees organized in a chunk grid,
/// providing seamless access to large voxel worlds.
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelWorld {
    /// Chunk size (2^depth)
    chunk_depth: u8,
    /// Loaded chunks
    chunks: std::collections::HashMap<(i32, i32, i32), Gd<VoxelTree>>,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelWorld {
    /// Creates a new voxel world with the specified chunk depth.
    ///
    /// # Arguments
    /// * `chunk_depth` - Depth of each chunk (5 = 32³, 6 = 64³)
    #[func]
    pub fn create(chunk_depth: i32) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            chunk_depth: chunk_depth.clamp(3, 7) as u8,
            chunks: std::collections::HashMap::new(),
            base,
        })
    }

    /// Gets the chunk size (voxels per axis).
    #[func]
    pub fn get_chunk_size(&self) -> i32 {
        1 << self.chunk_depth
    }

    /// Converts a world position to chunk coordinates.
    #[func]
    pub fn world_to_chunk(&self, world_pos: Vector3i) -> Vector3i {
        let chunk_size = 1 << self.chunk_depth;
        Vector3i::new(
            world_pos.x.div_euclid(chunk_size),
            world_pos.y.div_euclid(chunk_size),
            world_pos.z.div_euclid(chunk_size),
        )
    }

    /// Converts a world position to local chunk coordinates.
    #[func]
    pub fn world_to_local(&self, world_pos: Vector3i) -> Vector3i {
        let chunk_size = 1 << self.chunk_depth;
        Vector3i::new(
            world_pos.x.rem_euclid(chunk_size),
            world_pos.y.rem_euclid(chunk_size),
            world_pos.z.rem_euclid(chunk_size),
        )
    }

    /// Gets or creates a chunk at the specified chunk coordinates.
    #[func]
    pub fn get_or_create_chunk(&mut self, chunk_pos: Vector3i) -> Gd<VoxelTree> {
        let key = (chunk_pos.x, chunk_pos.y, chunk_pos.z);

        if let Some(chunk) = self.chunks.get(&key) {
            return chunk.clone();
        }

        let chunk = VoxelTree::create(self.chunk_depth as i32);
        self.chunks.insert(key, chunk.clone());
        chunk
    }

    /// Gets a voxel at world coordinates.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `world_pos` - World position
    #[func]
    pub fn get_voxel(&mut self, interner: Gd<VoxelInterner>, world_pos: Vector3i) -> i32 {
        let chunk_pos = self.world_to_chunk(world_pos);
        let local_pos = self.world_to_local(world_pos);

        let key = (chunk_pos.x, chunk_pos.y, chunk_pos.z);
        if let Some(chunk) = self.chunks.get(&key) {
            chunk.bind().get_voxel(interner, local_pos)
        } else {
            0 // Empty chunk
        }
    }

    /// Sets a voxel at world coordinates.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `world_pos` - World position
    /// * `value` - Voxel value
    #[func]
    pub fn set_voxel(&mut self, interner: Gd<VoxelInterner>, world_pos: Vector3i, value: i32) {
        let chunk_pos = self.world_to_chunk(world_pos);
        let local_pos = self.world_to_local(world_pos);

        let mut chunk = self.get_or_create_chunk(chunk_pos);
        chunk.bind_mut().set_voxel(interner, local_pos, value);
    }

    /// Returns the number of loaded chunks.
    #[func]
    pub fn get_chunk_count(&self) -> i32 {
        self.chunks.len() as i32
    }

    /// Unloads a chunk at the specified chunk coordinates.
    #[func]
    pub fn unload_chunk(&mut self, chunk_pos: Vector3i) -> bool {
        let key = (chunk_pos.x, chunk_pos.y, chunk_pos.z);
        self.chunks.remove(&key).is_some()
    }

    /// Returns all loaded chunk positions.
    #[func]
    pub fn get_loaded_chunks(&self) -> Array<Vector3i> {
        let mut arr = Array::new();
        for (x, y, z) in self.chunks.keys() {
            arr.push(Vector3i::new(*x, *y, *z));
        }
        arr
    }
}

// ============================================================================
// VoxelMeshBuilder - Converts voxel data to renderable meshes
// ============================================================================

/// Face direction for voxel mesh generation.
#[derive(Clone, Copy, Debug)]
enum VoxelFace {
    /// +X direction (right)
    Right,
    /// -X direction (left)
    Left,
    /// +Y direction (up)
    Top,
    /// -Y direction (down)
    Bottom,
    /// +Z direction (front)
    Front,
    /// -Z direction (back)
    Back,
}

impl VoxelFace {
    /// Returns the normal vector for this face.
    fn normal(&self) -> Vector3 {
        match self {
            VoxelFace::Right => Vector3::new(1.0, 0.0, 0.0),
            VoxelFace::Left => Vector3::new(-1.0, 0.0, 0.0),
            VoxelFace::Top => Vector3::new(0.0, 1.0, 0.0),
            VoxelFace::Bottom => Vector3::new(0.0, -1.0, 0.0),
            VoxelFace::Front => Vector3::new(0.0, 0.0, 1.0),
            VoxelFace::Back => Vector3::new(0.0, 0.0, -1.0),
        }
    }

    /// Returns the offset to check for neighbor in this direction.
    fn offset(&self) -> IVec3 {
        match self {
            VoxelFace::Right => IVec3::new(1, 0, 0),
            VoxelFace::Left => IVec3::new(-1, 0, 0),
            VoxelFace::Top => IVec3::new(0, 1, 0),
            VoxelFace::Bottom => IVec3::new(0, -1, 0),
            VoxelFace::Front => IVec3::new(0, 0, 1),
            VoxelFace::Back => IVec3::new(0, 0, -1),
        }
    }

    /// Returns the 4 vertices for this face (counter-clockwise when viewed from outside).
    fn vertices(&self, pos: Vector3, size: f32) -> [Vector3; 4] {
        let s = size;
        match self {
            VoxelFace::Right => [
                pos + Vector3::new(s, 0.0, 0.0),
                pos + Vector3::new(s, 0.0, s),
                pos + Vector3::new(s, s, s),
                pos + Vector3::new(s, s, 0.0),
            ],
            VoxelFace::Left => [
                pos + Vector3::new(0.0, 0.0, s),
                pos + Vector3::new(0.0, 0.0, 0.0),
                pos + Vector3::new(0.0, s, 0.0),
                pos + Vector3::new(0.0, s, s),
            ],
            VoxelFace::Top => [
                pos + Vector3::new(0.0, s, 0.0),
                pos + Vector3::new(s, s, 0.0),
                pos + Vector3::new(s, s, s),
                pos + Vector3::new(0.0, s, s),
            ],
            VoxelFace::Bottom => [
                pos + Vector3::new(0.0, 0.0, s),
                pos + Vector3::new(s, 0.0, s),
                pos + Vector3::new(s, 0.0, 0.0),
                pos + Vector3::new(0.0, 0.0, 0.0),
            ],
            VoxelFace::Front => [
                pos + Vector3::new(s, 0.0, s),
                pos + Vector3::new(0.0, 0.0, s),
                pos + Vector3::new(0.0, s, s),
                pos + Vector3::new(s, s, s),
            ],
            VoxelFace::Back => [
                pos + Vector3::new(0.0, 0.0, 0.0),
                pos + Vector3::new(s, 0.0, 0.0),
                pos + Vector3::new(s, s, 0.0),
                pos + Vector3::new(0.0, s, 0.0),
            ],
        }
    }

    /// Returns UV coordinates for this face.
    fn uvs(&self) -> [Vector2; 4] {
        [
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 0.0),
        ]
    }
}

const ALL_FACES: [VoxelFace; 6] = [
    VoxelFace::Right,
    VoxelFace::Left,
    VoxelFace::Top,
    VoxelFace::Bottom,
    VoxelFace::Front,
    VoxelFace::Back,
];

/// Builder for converting voxel data into renderable Godot meshes.
///
/// VoxelMeshBuilder generates optimized meshes from VoxelTree data by only
/// creating faces for voxels that are visible (not completely surrounded by
/// other solid voxels).
///
/// # Example
/// ```gdscript
/// var mesh_builder = VoxelMeshBuilder.new()
/// mesh_builder.set_voxel_size(1.0)
///
/// # Build mesh from a VoxelTree
/// var mesh = mesh_builder.build_mesh(interner, tree)
///
/// # Create MeshInstance3D to display it
/// var mesh_instance = MeshInstance3D.new()
/// mesh_instance.mesh = mesh
/// add_child(mesh_instance)
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelMeshBuilder {
    /// Size of each voxel in world units
    voxel_size: f32,
    /// Whether to generate vertex colors based on voxel type
    use_vertex_colors: bool,
    /// Color palette for voxel types (index = voxel type)
    color_palette: Vec<Color>,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelMeshBuilder {
    /// Creates a new VoxelMeshBuilder with default settings.
    #[func]
    pub fn create() -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            voxel_size: 1.0,
            use_vertex_colors: true,
            color_palette: Self::default_palette(),
            base,
        })
    }

    /// Sets the size of each voxel in world units.
    #[func]
    pub fn set_voxel_size(&mut self, size: f32) {
        self.voxel_size = size.max(0.001);
    }

    /// Gets the current voxel size.
    #[func]
    pub fn get_voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Enables or disables vertex colors based on voxel type.
    #[func]
    pub fn set_use_vertex_colors(&mut self, enabled: bool) {
        self.use_vertex_colors = enabled;
    }

    /// Sets a color for a specific voxel type in the palette.
    ///
    /// # Arguments
    /// * `voxel_type` - The voxel type (1-65535, 0 is empty)
    /// * `color` - The color to use for this voxel type
    #[func]
    pub fn set_voxel_color(&mut self, voxel_type: i32, color: Color) {
        let idx = voxel_type.max(0) as usize;
        if idx >= self.color_palette.len() {
            self.color_palette.resize(idx + 1, Color::WHITE);
        }
        self.color_palette[idx] = color;
    }

    /// Gets the color for a specific voxel type.
    #[func]
    pub fn get_voxel_color(&self, voxel_type: i32) -> Color {
        let idx = voxel_type.max(0) as usize;
        self.color_palette
            .get(idx)
            .copied()
            .unwrap_or(Color::WHITE)
    }

    /// Sets the entire color palette from an array.
    #[func]
    pub fn set_color_palette(&mut self, colors: Array<Color>) {
        self.color_palette = colors.iter_shared().collect();
    }

    /// Creates a default color palette with common voxel colors.
    fn default_palette() -> Vec<Color> {
        vec![
            Color::from_rgba(0.0, 0.0, 0.0, 0.0),     // 0: Empty (transparent)
            Color::from_rgba(0.5, 0.5, 0.5, 1.0),     // 1: Stone (gray)
            Color::from_rgba(0.4, 0.25, 0.13, 1.0),   // 2: Dirt (brown)
            Color::from_rgba(0.2, 0.6, 0.2, 1.0),     // 3: Grass (green)
            Color::from_rgba(0.6, 0.5, 0.3, 1.0),     // 4: Sand (tan)
            Color::from_rgba(0.3, 0.5, 0.8, 1.0),     // 5: Water (blue)
            Color::from_rgba(0.5, 0.35, 0.2, 1.0),    // 6: Wood (brown)
            Color::from_rgba(0.3, 0.5, 0.3, 1.0),     // 7: Leaves (dark green)
            Color::from_rgba(0.9, 0.9, 0.9, 1.0),     // 8: Snow (white)
            Color::from_rgba(0.8, 0.4, 0.1, 1.0),     // 9: Copper (orange)
            Color::from_rgba(0.8, 0.8, 0.2, 1.0),     // 10: Gold (yellow)
            Color::from_rgba(0.6, 0.6, 0.7, 1.0),     // 11: Iron (silver)
            Color::from_rgba(0.2, 0.2, 0.2, 1.0),     // 12: Coal (dark gray)
            Color::from_rgba(0.4, 0.1, 0.1, 1.0),     // 13: Redstone (red)
            Color::from_rgba(0.1, 0.5, 0.8, 1.0),     // 14: Diamond (cyan)
            Color::from_rgba(0.5, 0.2, 0.6, 1.0),     // 15: Amethyst (purple)
        ]
    }

    /// Builds a mesh from a VoxelTree using culled face generation.
    ///
    /// This method only generates faces for voxels that are visible (not
    /// completely surrounded by other solid voxels), which significantly
    /// reduces the polygon count.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `tree` - The VoxelTree to generate a mesh from
    ///
    /// # Returns
    /// An ArrayMesh ready to be used with MeshInstance3D.
    #[func]
    pub fn build_mesh(&self, interner: Gd<VoxelInterner>, tree: Gd<VoxelTree>) -> Gd<ArrayMesh> {
        let mut mesh = ArrayMesh::new_gd();

        let tree_bind = tree.bind();
        let Some(ref vox_tree) = tree_bind.tree else {
            godot_error!("VoxelMeshBuilder::build_mesh: VoxelTree not initialized");
            return mesh;
        };

        if vox_tree.get_root_id().is_empty() {
            // Empty tree, return empty mesh
            return mesh;
        }

        let interner_arc = interner.bind().get_arc();
        let interner_guard = interner_arc.read().unwrap();

        let resolution = 1i32 << tree_bind.max_depth;
        let size = self.voxel_size;

        // Collect mesh data
        let mut vertices: Vec<Vector3> = Vec::new();
        let mut normals: Vec<Vector3> = Vec::new();
        let mut uvs: Vec<Vector2> = Vec::new();
        let mut colors: Vec<Color> = Vec::new();
        let mut indices: Vec<i32> = Vec::new();

        // Iterate through all voxels
        for x in 0..resolution {
            for y in 0..resolution {
                for z in 0..resolution {
                    let pos = IVec3::new(x, y, z);
                    let voxel = vox_tree.get(&interner_guard, pos);

                    // Skip empty voxels
                    let voxel_type = match voxel {
                        Some(v) if v > 0 => v,
                        _ => continue,
                    };

                    let world_pos =
                        Vector3::new(x as f32 * size, y as f32 * size, z as f32 * size);
                    let color = self.get_voxel_color(voxel_type as i32);

                    // Check each face
                    for face in ALL_FACES {
                        let neighbor_pos = pos + face.offset();

                        // Check if neighbor is outside bounds or empty
                        let neighbor_empty = if neighbor_pos.x < 0
                            || neighbor_pos.x >= resolution
                            || neighbor_pos.y < 0
                            || neighbor_pos.y >= resolution
                            || neighbor_pos.z < 0
                            || neighbor_pos.z >= resolution
                        {
                            true
                        } else {
                            match vox_tree.get(&interner_guard, neighbor_pos) {
                                Some(v) if v > 0 => false,
                                _ => true,
                            }
                        };

                        // Only generate face if neighbor is empty
                        if neighbor_empty {
                            let base_idx = vertices.len() as i32;
                            let face_verts = face.vertices(world_pos, size);
                            let face_uvs = face.uvs();
                            let normal = face.normal();

                            // Add 4 vertices for this face
                            for i in 0..4 {
                                vertices.push(face_verts[i]);
                                normals.push(normal);
                                uvs.push(face_uvs[i]);
                                colors.push(color);
                            }

                            // Add 2 triangles (6 indices)
                            indices.push(base_idx);
                            indices.push(base_idx + 1);
                            indices.push(base_idx + 2);
                            indices.push(base_idx);
                            indices.push(base_idx + 2);
                            indices.push(base_idx + 3);
                        }
                    }
                }
            }
        }

        if vertices.is_empty() {
            return mesh;
        }

        // Convert to packed arrays
        let mut vert_arr = PackedVector3Array::new();
        let mut norm_arr = PackedVector3Array::new();
        let mut uv_arr = PackedVector2Array::new();
        let mut color_arr = PackedColorArray::new();
        let mut idx_arr = PackedInt32Array::new();

        for v in &vertices {
            vert_arr.push(*v);
        }
        for n in &normals {
            norm_arr.push(*n);
        }
        for uv in &uvs {
            uv_arr.push(*uv);
        }
        for c in &colors {
            color_arr.push(*c);
        }
        for i in &indices {
            idx_arr.push(*i);
        }

        // Create mesh arrays using varray! macro
        // ArrayType::MAX gives us the size needed
        let array_size = ArrayType::MAX.ord() as usize;
        let mut arrays: VarArray = varray![];
        arrays.resize(array_size, &Variant::nil());

        arrays.set(ArrayType::VERTEX.ord() as usize, &vert_arr.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &norm_arr.to_variant());
        arrays.set(ArrayType::TEX_UV.ord() as usize, &uv_arr.to_variant());
        if self.use_vertex_colors {
            arrays.set(ArrayType::COLOR.ord() as usize, &color_arr.to_variant());
        }
        arrays.set(ArrayType::INDEX.ord() as usize, &idx_arr.to_variant());

        mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);

        mesh
    }

    /// Builds a mesh from a specific chunk in a VoxelWorld.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `world` - The VoxelWorld containing the chunk
    /// * `chunk_pos` - The chunk coordinates to generate a mesh for
    ///
    /// # Returns
    /// An ArrayMesh ready to be used with MeshInstance3D, or empty if chunk doesn't exist.
    #[func]
    pub fn build_chunk_mesh(
        &self,
        interner: Gd<VoxelInterner>,
        world: Gd<VoxelWorld>,
        chunk_pos: Vector3i,
    ) -> Gd<ArrayMesh> {
        let world_bind = world.bind();
        let key = (chunk_pos.x, chunk_pos.y, chunk_pos.z);

        if let Some(chunk) = world_bind.chunks.get(&key) {
            self.build_mesh(interner, chunk.clone())
        } else {
            ArrayMesh::new_gd()
        }
    }

    /// Returns statistics about what a mesh build would produce.
    ///
    /// Useful for debugging and optimization.
    #[func]
    pub fn get_mesh_stats(&self, interner: Gd<VoxelInterner>, tree: Gd<VoxelTree>) -> VarDictionary {
        let mut stats = VarDictionary::new();

        let tree_bind = tree.bind();
        let Some(ref vox_tree) = tree_bind.tree else {
            stats.set("error", "VoxelTree not initialized");
            return stats;
        };

        let interner_arc = interner.bind().get_arc();
        let interner_guard = interner_arc.read().unwrap();

        let resolution = 1i32 << tree_bind.max_depth;
        let mut solid_voxels = 0i64;
        let mut visible_faces = 0i64;

        for x in 0..resolution {
            for y in 0..resolution {
                for z in 0..resolution {
                    let pos = IVec3::new(x, y, z);
                    let voxel = vox_tree.get(&interner_guard, pos);

                    if let Some(v) = voxel {
                        if v > 0 {
                            solid_voxels += 1;

                            for face in ALL_FACES {
                                let neighbor_pos = pos + face.offset();
                                let neighbor_empty = if neighbor_pos.x < 0
                                    || neighbor_pos.x >= resolution
                                    || neighbor_pos.y < 0
                                    || neighbor_pos.y >= resolution
                                    || neighbor_pos.z < 0
                                    || neighbor_pos.z >= resolution
                                {
                                    true
                                } else {
                                    match vox_tree.get(&interner_guard, neighbor_pos) {
                                        Some(v) if v > 0 => false,
                                        _ => true,
                                    }
                                };

                                if neighbor_empty {
                                    visible_faces += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        stats.set("resolution", resolution);
        stats.set("total_possible_voxels", (resolution as i64).pow(3));
        stats.set("solid_voxels", solid_voxels);
        stats.set("visible_faces", visible_faces);
        stats.set("estimated_vertices", visible_faces * 4);
        stats.set("estimated_triangles", visible_faces * 2);

        stats
    }

    /// Creates a default material suitable for voxel meshes with vertex colors.
    ///
    /// This function requires the `codegen-full` feature to be enabled.
    #[cfg(feature = "codegen-full")]
    #[func]
    pub fn create_default_material() -> Gd<godot::classes::StandardMaterial3D> {
        use godot::classes::base_material_3d;
        let mut material = godot::classes::StandardMaterial3D::new_gd();
        material.set_flag(base_material_3d::Flags::ALBEDO_FROM_VERTEX_COLOR, true);
        material.set_shading_mode(base_material_3d::ShadingMode::PER_PIXEL);
        material
    }
}

// ============================================================================
// VoxelChunkRenderer - Automatic mesh management for VoxelWorld
// ============================================================================

/// Automatic mesh management for rendering VoxelWorld chunks.
///
/// VoxelChunkRenderer handles the creation, updating, and cleanup of
/// MeshInstance3D nodes for each chunk in a VoxelWorld.
///
/// # Example
/// ```gdscript
/// extends Node3D
///
/// var interner: VoxelInterner
/// var world: VoxelWorld
/// var renderer: VoxelChunkRenderer
///
/// func _ready():
///     interner = VoxelInterner.new()
///     world = VoxelWorld.create(5)
///     renderer = VoxelChunkRenderer.new()
///     renderer.set_world(world)
///     renderer.set_interner(interner)
///     
///     # Modify voxels and mark chunks dirty
///     world.set_voxel(interner, Vector3i(0, 0, 0), 1)
///     renderer.mark_chunk_dirty(Vector3i(0, 0, 0))
///     
///     # Build mesh for the dirty chunk
///     var mesh = renderer.build_chunk(Vector3i(0, 0, 0))
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelChunkRenderer {
    /// The VoxelWorld to render
    world: Option<Gd<VoxelWorld>>,
    /// The VoxelInterner for memory
    interner: Option<Gd<VoxelInterner>>,
    /// Mesh builder for generating meshes
    mesh_builder: Gd<VoxelMeshBuilder>,
    /// Dirty chunks that need re-meshing
    dirty_chunks: std::collections::HashSet<(i32, i32, i32)>,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelChunkRenderer {
    /// Creates a new VoxelChunkRenderer.
    #[func]
    pub fn create() -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            world: None,
            interner: None,
            mesh_builder: VoxelMeshBuilder::create(),
            dirty_chunks: std::collections::HashSet::new(),
            base,
        })
    }

    /// Sets the VoxelWorld to render.
    #[func]
    pub fn set_world(&mut self, world: Gd<VoxelWorld>) {
        self.world = Some(world);
    }

    /// Sets the VoxelInterner for memory management.
    #[func]
    pub fn set_interner(&mut self, interner: Gd<VoxelInterner>) {
        self.interner = Some(interner);
    }

    /// Gets the mesh builder for customization.
    #[func]
    pub fn get_mesh_builder(&self) -> Gd<VoxelMeshBuilder> {
        self.mesh_builder.clone()
    }

    /// Sets a custom mesh builder.
    #[func]
    pub fn set_mesh_builder(&mut self, builder: Gd<VoxelMeshBuilder>) {
        self.mesh_builder = builder;
    }

    /// Marks a chunk as dirty (needs re-meshing).
    ///
    /// # Arguments
    /// * `chunk_pos` - The chunk coordinates to mark dirty
    #[func]
    pub fn mark_chunk_dirty(&mut self, chunk_pos: Vector3i) {
        self.dirty_chunks
            .insert((chunk_pos.x, chunk_pos.y, chunk_pos.z));
    }

    /// Marks the chunk containing a world position as dirty.
    #[func]
    pub fn mark_dirty_at(&mut self, world_pos: Vector3i) {
        if let Some(ref world) = self.world {
            let chunk_pos = world.bind().world_to_chunk(world_pos);
            self.mark_chunk_dirty(chunk_pos);
        }
    }

    /// Marks all loaded chunks as dirty.
    #[func]
    pub fn mark_all_dirty(&mut self) {
        if let Some(ref world) = self.world {
            let loaded = world.bind().get_loaded_chunks();
            for chunk_pos in loaded.iter_shared() {
                self.dirty_chunks
                    .insert((chunk_pos.x, chunk_pos.y, chunk_pos.z));
            }
        }
    }

    /// Returns the number of dirty chunks pending update.
    #[func]
    pub fn get_dirty_count(&self) -> i32 {
        self.dirty_chunks.len() as i32
    }

    /// Clears the dirty chunk list.
    #[func]
    pub fn clear_dirty(&mut self) {
        self.dirty_chunks.clear();
    }

    /// Returns whether there are any dirty chunks.
    #[func]
    pub fn has_dirty_chunks(&self) -> bool {
        !self.dirty_chunks.is_empty()
    }

    /// Gets the list of dirty chunk positions.
    #[func]
    pub fn get_dirty_chunks(&self) -> Array<Vector3i> {
        let mut arr = Array::new();
        for (x, y, z) in &self.dirty_chunks {
            arr.push(Vector3i::new(*x, *y, *z));
        }
        arr
    }

    /// Builds mesh for a single chunk. Returns the ArrayMesh or null if world/interner not set.
    #[func]
    pub fn build_chunk(&self, chunk_pos: Vector3i) -> Option<Gd<ArrayMesh>> {
        let world = self.world.as_ref()?;
        let interner = self.interner.as_ref()?;

        Some(
            self.mesh_builder
                .bind()
                .build_chunk_mesh(interner.clone(), world.clone(), chunk_pos),
        )
    }
}
