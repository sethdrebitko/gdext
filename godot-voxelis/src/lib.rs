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

// ============================================================================
// VoxelAsset - Reusable voxel structures/prefabs
// ============================================================================

/// A stored voxel entry with position and type.
#[derive(Clone, Debug)]
struct VoxelEntry {
    x: i32,
    y: i32,
    z: i32,
    voxel_type: u16,
}

/// A reusable voxel structure that can be placed in trees or worlds.
///
/// VoxelAsset stores a collection of voxels that can be saved, loaded,
/// and placed multiple times. Use this for prefabs like trees, rocks,
/// buildings, furniture, etc.
///
/// # Example
/// ```gdscript
/// # Create an asset manually
/// var asset = VoxelAsset.new()
/// asset.set_voxel(Vector3i(0, 0, 0), 1)  # Base
/// asset.set_voxel(Vector3i(0, 1, 0), 1)  # Trunk
/// asset.set_voxel(Vector3i(0, 2, 0), 2)  # Leaves
///
/// # Or generate procedurally
/// var tree = VoxelAssetGenerator.generate_tree(5, 3)
///
/// # Place in world
/// tree.place_in_world(interner, world, Vector3i(10, 0, 10))
///
/// # Save/load
/// var data = asset.to_dictionary()
/// var loaded = VoxelAsset.from_dictionary(data)
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelAsset {
    /// Stored voxels
    voxels: Vec<VoxelEntry>,
    /// Asset name
    name: GString,
    /// Bounding box min
    bounds_min: Vector3i,
    /// Bounding box max
    bounds_max: Vector3i,
    /// Whether bounds need recalculation
    bounds_dirty: bool,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelAsset {
    /// Creates a new empty VoxelAsset.
    #[func]
    pub fn create() -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            voxels: Vec::new(),
            name: GString::new(),
            bounds_min: Vector3i::ZERO,
            bounds_max: Vector3i::ZERO,
            bounds_dirty: true,
            base,
        })
    }

    /// Creates a new VoxelAsset with the given name.
    #[func]
    pub fn create_named(name: GString) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            voxels: Vec::new(),
            name,
            bounds_min: Vector3i::ZERO,
            bounds_max: Vector3i::ZERO,
            bounds_dirty: true,
            base,
        })
    }

    /// Gets the asset name.
    #[func]
    pub fn get_name(&self) -> GString {
        self.name.clone()
    }

    /// Sets the asset name.
    #[func]
    pub fn set_name(&mut self, name: GString) {
        self.name = name;
    }

    /// Sets a voxel in the asset at the given local position.
    #[func]
    pub fn set_voxel(&mut self, position: Vector3i, voxel_type: i32) {
        // Remove existing voxel at this position
        self.voxels
            .retain(|v| v.x != position.x || v.y != position.y || v.z != position.z);

        // Add new voxel if not empty
        if voxel_type > 0 {
            self.voxels.push(VoxelEntry {
                x: position.x,
                y: position.y,
                z: position.z,
                voxel_type: voxel_type.clamp(0, 65535) as u16,
            });
        }

        self.bounds_dirty = true;
    }

    /// Gets the voxel type at the given local position.
    #[func]
    pub fn get_voxel(&self, position: Vector3i) -> i32 {
        for v in &self.voxels {
            if v.x == position.x && v.y == position.y && v.z == position.z {
                return v.voxel_type as i32;
            }
        }
        0
    }

    /// Removes a voxel at the given position.
    #[func]
    pub fn remove_voxel(&mut self, position: Vector3i) {
        self.voxels
            .retain(|v| v.x != position.x || v.y != position.y || v.z != position.z);
        self.bounds_dirty = true;
    }

    /// Clears all voxels from the asset.
    #[func]
    pub fn clear(&mut self) {
        self.voxels.clear();
        self.bounds_dirty = true;
    }

    /// Returns the number of voxels in the asset.
    #[func]
    pub fn get_voxel_count(&self) -> i32 {
        self.voxels.len() as i32
    }

    /// Returns whether the asset is empty.
    #[func]
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Recalculates the bounding box.
    fn recalculate_bounds(&mut self) {
        if !self.bounds_dirty {
            return;
        }

        if self.voxels.is_empty() {
            self.bounds_min = Vector3i::ZERO;
            self.bounds_max = Vector3i::ZERO;
        } else {
            let mut min_x = i32::MAX;
            let mut min_y = i32::MAX;
            let mut min_z = i32::MAX;
            let mut max_x = i32::MIN;
            let mut max_y = i32::MIN;
            let mut max_z = i32::MIN;

            for v in &self.voxels {
                min_x = min_x.min(v.x);
                min_y = min_y.min(v.y);
                min_z = min_z.min(v.z);
                max_x = max_x.max(v.x);
                max_y = max_y.max(v.y);
                max_z = max_z.max(v.z);
            }

            self.bounds_min = Vector3i::new(min_x, min_y, min_z);
            self.bounds_max = Vector3i::new(max_x, max_y, max_z);
        }

        self.bounds_dirty = false;
    }

    /// Gets the minimum corner of the bounding box.
    #[func]
    pub fn get_bounds_min(&mut self) -> Vector3i {
        self.recalculate_bounds();
        self.bounds_min
    }

    /// Gets the maximum corner of the bounding box.
    #[func]
    pub fn get_bounds_max(&mut self) -> Vector3i {
        self.recalculate_bounds();
        self.bounds_max
    }

    /// Gets the size of the asset (max - min + 1).
    #[func]
    pub fn get_size(&mut self) -> Vector3i {
        self.recalculate_bounds();
        if self.voxels.is_empty() {
            Vector3i::ZERO
        } else {
            Vector3i::new(
                self.bounds_max.x - self.bounds_min.x + 1,
                self.bounds_max.y - self.bounds_min.y + 1,
                self.bounds_max.z - self.bounds_min.z + 1,
            )
        }
    }

    /// Centers the asset around the origin (0, 0, 0).
    #[func]
    pub fn center(&mut self) {
        self.recalculate_bounds();
        if self.voxels.is_empty() {
            return;
        }

        let center_x = (self.bounds_min.x + self.bounds_max.x) / 2;
        let center_z = (self.bounds_min.z + self.bounds_max.z) / 2;

        for v in &mut self.voxels {
            v.x -= center_x;
            v.z -= center_z;
        }

        self.bounds_dirty = true;
    }

    /// Moves the asset so its minimum Y is at the origin.
    #[func]
    pub fn ground(&mut self) {
        self.recalculate_bounds();
        if self.voxels.is_empty() {
            return;
        }

        let min_y = self.bounds_min.y;
        for v in &mut self.voxels {
            v.y -= min_y;
        }

        self.bounds_dirty = true;
    }

    /// Translates all voxels by the given offset.
    #[func]
    pub fn translate(&mut self, offset: Vector3i) {
        for v in &mut self.voxels {
            v.x += offset.x;
            v.y += offset.y;
            v.z += offset.z;
        }
        self.bounds_dirty = true;
    }

    /// Rotates the asset 90 degrees around the Y axis.
    #[func]
    pub fn rotate_y_90(&mut self) {
        for v in &mut self.voxels {
            let old_x = v.x;
            v.x = -v.z;
            v.z = old_x;
        }
        self.bounds_dirty = true;
    }

    /// Rotates the asset 180 degrees around the Y axis.
    #[func]
    pub fn rotate_y_180(&mut self) {
        for v in &mut self.voxels {
            v.x = -v.x;
            v.z = -v.z;
        }
        self.bounds_dirty = true;
    }

    /// Rotates the asset 270 degrees around the Y axis.
    #[func]
    pub fn rotate_y_270(&mut self) {
        for v in &mut self.voxels {
            let old_x = v.x;
            v.x = v.z;
            v.z = -old_x;
        }
        self.bounds_dirty = true;
    }

    /// Mirrors the asset along the X axis.
    #[func]
    pub fn mirror_x(&mut self) {
        for v in &mut self.voxels {
            v.x = -v.x;
        }
        self.bounds_dirty = true;
    }

    /// Mirrors the asset along the Z axis.
    #[func]
    pub fn mirror_z(&mut self) {
        for v in &mut self.voxels {
            v.z = -v.z;
        }
        self.bounds_dirty = true;
    }

    /// Places the asset into a VoxelTree at the given position.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `tree` - The target VoxelTree
    /// * `position` - The position to place the asset (added to each voxel's local position)
    #[func]
    pub fn place_in_tree(
        &self,
        interner: Gd<VoxelInterner>,
        mut tree: Gd<VoxelTree>,
        position: Vector3i,
    ) {
        let mut tree_bind = tree.bind_mut();
        let max_depth = tree_bind.max_depth;
        let Some(ref mut vox_tree) = tree_bind.tree else {
            godot_error!("VoxelAsset::place_in_tree: VoxelTree not initialized");
            return;
        };

        let interner_arc = interner.bind().get_arc();
        let mut interner_guard = interner_arc.write().unwrap();

        let max = 1 << max_depth;
        for v in &self.voxels {
            let pos = IVec3::new(position.x + v.x, position.y + v.y, position.z + v.z);

            if pos.x >= 0 && pos.x < max && pos.y >= 0 && pos.y < max && pos.z >= 0 && pos.z < max {
                vox_tree.set(&mut interner_guard, pos, v.voxel_type);
            }
        }
    }

    /// Places the asset into a VoxelWorld at the given world position.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `world` - The target VoxelWorld
    /// * `position` - The world position to place the asset
    #[func]
    pub fn place_in_world(
        &self,
        interner: Gd<VoxelInterner>,
        mut world: Gd<VoxelWorld>,
        position: Vector3i,
    ) {
        for v in &self.voxels {
            let world_pos = Vector3i::new(position.x + v.x, position.y + v.y, position.z + v.z);
            world
                .bind_mut()
                .set_voxel(interner.clone(), world_pos, v.voxel_type as i32);
        }
    }

    /// Places the asset using a VoxelBatch for better performance.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `batch` - The batch to add operations to
    /// * `position` - The position to place the asset
    #[func]
    pub fn place_in_batch(
        &self,
        interner: Gd<VoxelInterner>,
        mut batch: Gd<VoxelBatch>,
        position: Vector3i,
    ) {
        for v in &self.voxels {
            let pos = Vector3i::new(position.x + v.x, position.y + v.y, position.z + v.z);
            batch
                .bind_mut()
                .set_voxel(interner.clone(), pos, v.voxel_type as i32);
        }
    }

    /// Converts the asset to a Dictionary for serialization.
    #[func]
    pub fn to_dictionary(&mut self) -> VarDictionary {
        self.recalculate_bounds();

        let mut dict = VarDictionary::new();
        dict.set("name", self.name.clone());
        dict.set("version", 1i32);

        // Store voxels as packed arrays for efficiency
        let mut positions = PackedInt32Array::new();
        let mut types = PackedInt32Array::new();

        for v in &self.voxels {
            positions.push(v.x);
            positions.push(v.y);
            positions.push(v.z);
            types.push(v.voxel_type as i32);
        }

        dict.set("positions", positions);
        dict.set("types", types);
        dict.set("bounds_min", self.bounds_min);
        dict.set("bounds_max", self.bounds_max);

        dict
    }

    /// Creates a VoxelAsset from a Dictionary.
    #[func]
    pub fn from_dictionary(dict: VarDictionary) -> Option<Gd<Self>> {
        let name: GString = dict.get("name")?.to();
        let positions: PackedInt32Array = dict.get("positions")?.to();
        let types: PackedInt32Array = dict.get("types")?.to();

        if positions.len() % 3 != 0 || positions.len() / 3 != types.len() {
            godot_error!("VoxelAsset::from_dictionary: Invalid data format");
            return None;
        }

        let mut voxels = Vec::new();
        for i in 0..types.len() {
            voxels.push(VoxelEntry {
                x: positions[i * 3],
                y: positions[i * 3 + 1],
                z: positions[i * 3 + 2],
                voxel_type: types[i].clamp(0, 65535) as u16,
            });
        }

        Some(Gd::from_init_fn(|base| Self {
            voxels,
            name,
            bounds_min: Vector3i::ZERO,
            bounds_max: Vector3i::ZERO,
            bounds_dirty: true,
            base,
        }))
    }

    /// Creates a copy of this asset.
    #[func]
    pub fn duplicate(&self) -> Gd<Self> {
        Gd::from_init_fn(|base| Self {
            voxels: self.voxels.clone(),
            name: self.name.clone(),
            bounds_min: self.bounds_min,
            bounds_max: self.bounds_max,
            bounds_dirty: self.bounds_dirty,
            base,
        })
    }

    /// Gets all voxel positions as an array.
    #[func]
    pub fn get_positions(&self) -> Array<Vector3i> {
        let mut arr = Array::new();
        for v in &self.voxels {
            arr.push(Vector3i::new(v.x, v.y, v.z));
        }
        arr
    }

    /// Fills a box region with a voxel type.
    #[func]
    pub fn fill_box(&mut self, min_corner: Vector3i, max_corner: Vector3i, voxel_type: i32) {
        let min_x = min_corner.x.min(max_corner.x);
        let max_x = min_corner.x.max(max_corner.x);
        let min_y = min_corner.y.min(max_corner.y);
        let max_y = min_corner.y.max(max_corner.y);
        let min_z = min_corner.z.min(max_corner.z);
        let max_z = min_corner.z.max(max_corner.z);

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    self.set_voxel(Vector3i::new(x, y, z), voxel_type);
                }
            }
        }
    }

    /// Fills a sphere region with a voxel type.
    #[func]
    pub fn fill_sphere(&mut self, center: Vector3i, radius: i32, voxel_type: i32) {
        let radius_sq = radius * radius;

        for x in (center.x - radius)..=(center.x + radius) {
            for y in (center.y - radius)..=(center.y + radius) {
                for z in (center.z - radius)..=(center.z + radius) {
                    let dx = x - center.x;
                    let dy = y - center.y;
                    let dz = z - center.z;
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq <= radius_sq {
                        self.set_voxel(Vector3i::new(x, y, z), voxel_type);
                    }
                }
            }
        }
    }

    /// Fills a cylinder region with a voxel type.
    #[func]
    pub fn fill_cylinder(
        &mut self,
        base_center: Vector3i,
        radius: i32,
        height: i32,
        voxel_type: i32,
    ) {
        let radius_sq = radius * radius;

        for y in 0..height {
            for x in (base_center.x - radius)..=(base_center.x + radius) {
                for z in (base_center.z - radius)..=(base_center.z + radius) {
                    let dx = x - base_center.x;
                    let dz = z - base_center.z;
                    let dist_sq = dx * dx + dz * dz;

                    if dist_sq <= radius_sq {
                        self.set_voxel(Vector3i::new(x, base_center.y + y, z), voxel_type);
                    }
                }
            }
        }
    }
}

// ============================================================================
// VoxelAssetGenerator - Procedural asset generation
// ============================================================================

/// Procedural generator for common voxel assets.
///
/// Use this to quickly create trees, rocks, buildings, and other
/// common structures.
///
/// # Example
/// ```gdscript
/// # Generate a tree
/// var tree = VoxelAssetGenerator.generate_tree(5, 3)
/// tree.place_in_world(interner, world, Vector3i(10, 0, 10))
///
/// # Generate a rock
/// var rock = VoxelAssetGenerator.generate_rock(3)
/// rock.place_in_world(interner, world, Vector3i(20, 0, 15))
///
/// # Generate a simple house
/// var house = VoxelAssetGenerator.generate_house(8, 6, 8)
/// house.place_in_world(interner, world, Vector3i(30, 0, 20))
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelAssetGenerator {
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelAssetGenerator {
    // Voxel type constants for generated assets
    // These match the default color palette in VoxelMeshBuilder
    const STONE: i32 = 1;
    const DIRT: i32 = 2;
    const GRASS: i32 = 3;
    const WOOD: i32 = 6;
    const LEAVES: i32 = 7;

    /// Generates a simple tree asset.
    ///
    /// # Arguments
    /// * `trunk_height` - Height of the trunk (3-20)
    /// * `canopy_radius` - Radius of the leaf canopy (2-10)
    ///
    /// # Returns
    /// A VoxelAsset containing the tree, centered at base.
    #[func]
    pub fn generate_tree(trunk_height: i32, canopy_radius: i32) -> Gd<VoxelAsset> {
        let trunk_height = trunk_height.clamp(3, 20);
        let canopy_radius = canopy_radius.clamp(2, 10);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Tree".into());

        // Generate trunk
        for y in 0..trunk_height {
            asset_bind.set_voxel(Vector3i::new(0, y, 0), Self::WOOD);
        }

        // Generate canopy (sphere of leaves)
        let canopy_center_y = trunk_height + canopy_radius / 2;
        let radius_sq = canopy_radius * canopy_radius;

        for x in -canopy_radius..=canopy_radius {
            for y in -canopy_radius..=canopy_radius {
                for z in -canopy_radius..=canopy_radius {
                    let dist_sq = x * x + y * y + z * z;
                    if dist_sq <= radius_sq {
                        let pos = Vector3i::new(x, canopy_center_y + y, z);
                        // Don't overwrite trunk
                        if !(x == 0 && z == 0 && pos.y < trunk_height) {
                            asset_bind.set_voxel(pos, Self::LEAVES);
                        }
                    }
                }
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a pine tree asset.
    ///
    /// # Arguments
    /// * `height` - Total height of the tree (5-30)
    ///
    /// # Returns
    /// A VoxelAsset containing the pine tree.
    #[func]
    pub fn generate_pine_tree(height: i32) -> Gd<VoxelAsset> {
        let height = height.clamp(5, 30);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Pine Tree".into());

        // Generate trunk
        let trunk_height = height * 2 / 3;
        for y in 0..trunk_height {
            asset_bind.set_voxel(Vector3i::new(0, y, 0), Self::WOOD);
        }

        // Generate conical canopy
        let canopy_start = height / 4;
        let canopy_height = height - canopy_start;

        for y in 0..canopy_height {
            let layer_y = canopy_start + y;
            let progress = y as f32 / canopy_height as f32;
            let radius = ((1.0 - progress) * (height as f32 / 4.0)).ceil() as i32;

            for x in -radius..=radius {
                for z in -radius..=radius {
                    let dist_sq = x * x + z * z;
                    if dist_sq <= radius * radius {
                        if !(x == 0 && z == 0 && layer_y < trunk_height) {
                            asset_bind.set_voxel(Vector3i::new(x, layer_y, z), Self::LEAVES);
                        }
                    }
                }
            }
        }

        // Top of tree
        asset_bind.set_voxel(Vector3i::new(0, height - 1, 0), Self::LEAVES);
        asset_bind.set_voxel(Vector3i::new(0, height, 0), Self::LEAVES);

        drop(asset_bind);
        asset
    }

    /// Generates a rock/boulder asset.
    ///
    /// # Arguments
    /// * `size` - Approximate size of the rock (2-10)
    ///
    /// # Returns
    /// A VoxelAsset containing the rock.
    #[func]
    pub fn generate_rock(size: i32) -> Gd<VoxelAsset> {
        let size = size.clamp(2, 10);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Rock".into());

        // Generate an irregular rock shape using multiple overlapping spheres
        let main_radius = size;
        let radius_sq = main_radius * main_radius;

        for x in -main_radius..=main_radius {
            for y in 0..=main_radius {
                for z in -main_radius..=main_radius {
                    let dist_sq = x * x + (y * 3 / 2) * (y * 3 / 2) + z * z;
                    if dist_sq <= radius_sq {
                        asset_bind.set_voxel(Vector3i::new(x, y, z), Self::STONE);
                    }
                }
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a simple house/building asset.
    ///
    /// # Arguments
    /// * `width` - Width of the house (X axis, 4-20)
    /// * `height` - Height to the roof base (4-15)
    /// * `depth` - Depth of the house (Z axis, 4-20)
    ///
    /// # Returns
    /// A VoxelAsset containing the house.
    #[func]
    pub fn generate_house(width: i32, height: i32, depth: i32) -> Gd<VoxelAsset> {
        let width = width.clamp(4, 20);
        let height = height.clamp(4, 15);
        let depth = depth.clamp(4, 20);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("House".into());

        // Generate walls
        for y in 0..height {
            for x in 0..width {
                for z in 0..depth {
                    let is_wall =
                        x == 0 || x == width - 1 || z == 0 || z == depth - 1 || y == 0;
                    if is_wall {
                        asset_bind.set_voxel(Vector3i::new(x, y, z), Self::STONE);
                    }
                }
            }
        }

        // Add door (front wall, centered)
        let door_x = width / 2;
        asset_bind.set_voxel(Vector3i::new(door_x, 1, 0), 0);
        asset_bind.set_voxel(Vector3i::new(door_x, 2, 0), 0);

        // Add windows
        if width > 6 {
            let window_y = height / 2;
            // Front windows
            asset_bind.set_voxel(Vector3i::new(2, window_y, 0), 0);
            asset_bind.set_voxel(Vector3i::new(width - 3, window_y, 0), 0);
            // Back windows
            asset_bind.set_voxel(Vector3i::new(2, window_y, depth - 1), 0);
            asset_bind.set_voxel(Vector3i::new(width - 3, window_y, depth - 1), 0);
        }

        // Generate simple roof
        for y in 0..=(width / 2) {
            for z in -1..=depth {
                let roof_y = height + y;
                asset_bind.set_voxel(Vector3i::new(y, roof_y, z), Self::WOOD);
                asset_bind.set_voxel(Vector3i::new(width - 1 - y, roof_y, z), Self::WOOD);
            }
        }

        // Center the house
        drop(asset_bind);
        let mut asset_bind = asset.bind_mut();
        asset_bind.translate(Vector3i::new(-width / 2, 0, -depth / 2));

        drop(asset_bind);
        asset
    }

    /// Generates a wall segment.
    ///
    /// # Arguments
    /// * `length` - Length of the wall
    /// * `height` - Height of the wall
    /// * `thickness` - Thickness of the wall (1-3)
    ///
    /// # Returns
    /// A VoxelAsset containing the wall, aligned along the X axis.
    #[func]
    pub fn generate_wall(length: i32, height: i32, thickness: i32) -> Gd<VoxelAsset> {
        let length = length.clamp(1, 100);
        let height = height.clamp(1, 50);
        let thickness = thickness.clamp(1, 3);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Wall".into());

        for x in 0..length {
            for y in 0..height {
                for z in 0..thickness {
                    asset_bind.set_voxel(Vector3i::new(x, y, z), Self::STONE);
                }
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a pillar/column.
    ///
    /// # Arguments
    /// * `height` - Height of the pillar
    /// * `radius` - Radius of the pillar (1-5)
    ///
    /// # Returns
    /// A VoxelAsset containing the pillar.
    #[func]
    pub fn generate_pillar(height: i32, radius: i32) -> Gd<VoxelAsset> {
        let height = height.clamp(1, 50);
        let radius = radius.clamp(1, 5);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Pillar".into());

        asset_bind.fill_cylinder(Vector3i::ZERO, radius, height, Self::STONE);

        drop(asset_bind);
        asset
    }

    /// Generates a staircase.
    ///
    /// # Arguments
    /// * `width` - Width of the stairs
    /// * `steps` - Number of steps
    ///
    /// # Returns
    /// A VoxelAsset containing the stairs, going up in +X direction.
    #[func]
    pub fn generate_stairs(width: i32, steps: i32) -> Gd<VoxelAsset> {
        let width = width.clamp(1, 10);
        let steps = steps.clamp(1, 20);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Stairs".into());

        for step in 0..steps {
            for z in 0..width {
                // Each step is 1 block high and 1 block deep
                for fill_y in 0..=step {
                    asset_bind.set_voxel(Vector3i::new(step, fill_y, z), Self::STONE);
                }
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a fence segment.
    ///
    /// # Arguments
    /// * `length` - Length of the fence
    /// * `height` - Height of the fence posts (2-5)
    ///
    /// # Returns
    /// A VoxelAsset containing the fence.
    #[func]
    pub fn generate_fence(length: i32, height: i32) -> Gd<VoxelAsset> {
        let length = length.clamp(1, 50);
        let height = height.clamp(2, 5);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Fence".into());

        // Posts every 3 blocks
        for x in (0..length).step_by(3) {
            for y in 0..height {
                asset_bind.set_voxel(Vector3i::new(x, y, 0), Self::WOOD);
            }
        }
        // End post
        for y in 0..height {
            asset_bind.set_voxel(Vector3i::new(length - 1, y, 0), Self::WOOD);
        }

        // Horizontal rails
        let rail_y = height / 2;
        for x in 0..length {
            asset_bind.set_voxel(Vector3i::new(x, rail_y, 0), Self::WOOD);
            asset_bind.set_voxel(Vector3i::new(x, height - 1, 0), Self::WOOD);
        }

        drop(asset_bind);
        asset
    }

    /// Generates a bush/shrub.
    ///
    /// # Arguments
    /// * `size` - Size of the bush (1-5)
    ///
    /// # Returns
    /// A VoxelAsset containing the bush.
    #[func]
    pub fn generate_bush(size: i32) -> Gd<VoxelAsset> {
        let size = size.clamp(1, 5);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Bush".into());

        // Slightly flattened sphere
        let radius_sq = size * size;
        for x in -size..=size {
            for y in 0..=size {
                for z in -size..=size {
                    let dist_sq = x * x + (y * 2) * (y * 2) + z * z;
                    if dist_sq <= radius_sq {
                        asset_bind.set_voxel(Vector3i::new(x, y, z), Self::LEAVES);
                    }
                }
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a cactus.
    ///
    /// # Arguments
    /// * `height` - Height of the cactus (3-15)
    ///
    /// # Returns
    /// A VoxelAsset containing the cactus.
    #[func]
    pub fn generate_cactus(height: i32) -> Gd<VoxelAsset> {
        let height = height.clamp(3, 15);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Cactus".into());

        // Main stem
        for y in 0..height {
            asset_bind.set_voxel(Vector3i::new(0, y, 0), Self::GRASS);
        }

        // Arms (if tall enough)
        if height >= 5 {
            let arm_y = height / 2;
            // Left arm
            asset_bind.set_voxel(Vector3i::new(-1, arm_y, 0), Self::GRASS);
            asset_bind.set_voxel(Vector3i::new(-1, arm_y + 1, 0), Self::GRASS);
            asset_bind.set_voxel(Vector3i::new(-1, arm_y + 2, 0), Self::GRASS);

            // Right arm (slightly higher)
            let arm_y2 = arm_y + 2;
            if arm_y2 + 2 < height {
                asset_bind.set_voxel(Vector3i::new(1, arm_y2, 0), Self::GRASS);
                asset_bind.set_voxel(Vector3i::new(1, arm_y2 + 1, 0), Self::GRASS);
                asset_bind.set_voxel(Vector3i::new(1, arm_y2 + 2, 0), Self::GRASS);
            }
        }

        drop(asset_bind);
        asset
    }

    /// Generates a flower bed with dirt base.
    ///
    /// # Arguments
    /// * `width` - Width of the bed
    /// * `depth` - Depth of the bed
    ///
    /// # Returns
    /// A VoxelAsset containing the flower bed.
    #[func]
    pub fn generate_flower_bed(width: i32, depth: i32) -> Gd<VoxelAsset> {
        let width = width.clamp(2, 20);
        let depth = depth.clamp(2, 20);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Flower Bed".into());

        // Dirt base
        for x in 0..width {
            for z in 0..depth {
                asset_bind.set_voxel(Vector3i::new(x, 0, z), Self::DIRT);
            }
        }

        // Some flowers on top (using leaves color as placeholder)
        for x in (1..width - 1).step_by(2) {
            for z in (1..depth - 1).step_by(2) {
                asset_bind.set_voxel(Vector3i::new(x, 1, z), Self::LEAVES);
            }
        }

        // Center the bed
        drop(asset_bind);
        let mut asset_bind = asset.bind_mut();
        asset_bind.translate(Vector3i::new(-width / 2, 0, -depth / 2));

        drop(asset_bind);
        asset
    }

    /// Generates a path segment.
    ///
    /// # Arguments
    /// * `length` - Length of the path
    /// * `width` - Width of the path (1-5)
    ///
    /// # Returns
    /// A VoxelAsset containing the path along the X axis.
    #[func]
    pub fn generate_path(length: i32, width: i32) -> Gd<VoxelAsset> {
        let length = length.clamp(1, 100);
        let width = width.clamp(1, 5);

        let mut asset = VoxelAsset::create();
        let mut asset_bind = asset.bind_mut();
        asset_bind.set_name("Path".into());

        let half_width = width / 2;
        for x in 0..length {
            for z in -half_width..=(width - half_width - 1) {
                asset_bind.set_voxel(Vector3i::new(x, 0, z), Self::DIRT);
            }
        }

        drop(asset_bind);
        asset
    }
}
