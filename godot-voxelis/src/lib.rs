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
//! # Rendering voxels
//! var mesher = VoxelMesher.new()
//! mesher.set_voxel_color(1, Color(0.5, 0.3, 0.1))  # Brown for dirt
//! mesher.set_voxel_color(2, Color(0.2, 0.8, 0.2))  # Green for grass
//! var mesh = mesher.generate_mesh(interner, tree)
//! $MeshInstance3D.mesh = mesh
//! ```

use std::sync::{Arc, RwLock};

use glam::IVec3;
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
// VoxelMesher - Mesh generation for voxel rendering
// ============================================================================

/// Face direction for mesh generation
#[derive(Clone, Copy, Debug)]
enum FaceDirection {
    Top,    // +Y
    Bottom, // -Y
    North,  // +Z
    South,  // -Z
    East,   // +X
    West,   // -X
}

impl FaceDirection {
    /// Returns the normal vector for this face direction
    fn normal(&self) -> Vector3 {
        match self {
            FaceDirection::Top => Vector3::new(0.0, 1.0, 0.0),
            FaceDirection::Bottom => Vector3::new(0.0, -1.0, 0.0),
            FaceDirection::North => Vector3::new(0.0, 0.0, 1.0),
            FaceDirection::South => Vector3::new(0.0, 0.0, -1.0),
            FaceDirection::East => Vector3::new(1.0, 0.0, 0.0),
            FaceDirection::West => Vector3::new(-1.0, 0.0, 0.0),
        }
    }

    /// Returns the offset to check for neighbor in this direction
    fn offset(&self) -> IVec3 {
        match self {
            FaceDirection::Top => IVec3::new(0, 1, 0),
            FaceDirection::Bottom => IVec3::new(0, -1, 0),
            FaceDirection::North => IVec3::new(0, 0, 1),
            FaceDirection::South => IVec3::new(0, 0, -1),
            FaceDirection::East => IVec3::new(1, 0, 0),
            FaceDirection::West => IVec3::new(-1, 0, 0),
        }
    }

    /// Returns the four vertices of a face at the given position (in counter-clockwise order for front face)
    fn vertices(&self, pos: IVec3, voxel_size: f32) -> [Vector3; 4] {
        let p = Vector3::new(
            pos.x as f32 * voxel_size,
            pos.y as f32 * voxel_size,
            pos.z as f32 * voxel_size,
        );
        let s = voxel_size;

        match self {
            FaceDirection::Top => [
                Vector3::new(p.x, p.y + s, p.z),
                Vector3::new(p.x, p.y + s, p.z + s),
                Vector3::new(p.x + s, p.y + s, p.z + s),
                Vector3::new(p.x + s, p.y + s, p.z),
            ],
            FaceDirection::Bottom => [
                Vector3::new(p.x, p.y, p.z + s),
                Vector3::new(p.x, p.y, p.z),
                Vector3::new(p.x + s, p.y, p.z),
                Vector3::new(p.x + s, p.y, p.z + s),
            ],
            FaceDirection::North => [
                Vector3::new(p.x + s, p.y, p.z + s),
                Vector3::new(p.x + s, p.y + s, p.z + s),
                Vector3::new(p.x, p.y + s, p.z + s),
                Vector3::new(p.x, p.y, p.z + s),
            ],
            FaceDirection::South => [
                Vector3::new(p.x, p.y, p.z),
                Vector3::new(p.x, p.y + s, p.z),
                Vector3::new(p.x + s, p.y + s, p.z),
                Vector3::new(p.x + s, p.y, p.z),
            ],
            FaceDirection::East => [
                Vector3::new(p.x + s, p.y, p.z),
                Vector3::new(p.x + s, p.y + s, p.z),
                Vector3::new(p.x + s, p.y + s, p.z + s),
                Vector3::new(p.x + s, p.y, p.z + s),
            ],
            FaceDirection::West => [
                Vector3::new(p.x, p.y, p.z + s),
                Vector3::new(p.x, p.y + s, p.z + s),
                Vector3::new(p.x, p.y + s, p.z),
                Vector3::new(p.x, p.y, p.z),
            ],
        }
    }

    /// Returns UV coordinates for a face (standard 0-1 mapping)
    fn uvs(&self) -> [Vector2; 4] {
        [
            Vector2::new(0.0, 1.0),
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        ]
    }
}

const ALL_DIRECTIONS: [FaceDirection; 6] = [
    FaceDirection::Top,
    FaceDirection::Bottom,
    FaceDirection::North,
    FaceDirection::South,
    FaceDirection::East,
    FaceDirection::West,
];

/// A mesh generator for voxel data.
///
/// VoxelMesher converts voxel data into renderable Godot meshes using culled
/// face generation (only visible faces between solid and empty voxels are rendered).
///
/// # Features
/// - Culled face generation for efficient rendering
/// - Per-voxel-type color support
/// - Configurable voxel size
/// - Optional greedy meshing for reduced polygon count
///
/// # Usage
/// ```gdscript
/// var mesher = VoxelMesher.new()
/// mesher.set_voxel_color(1, Color.BROWN)   # Stone
/// mesher.set_voxel_color(2, Color.GREEN)   # Grass
/// mesher.set_voxel_color(3, Color.BLUE)    # Water
///
/// var mesh = mesher.generate_mesh(interner, tree)
/// $MeshInstance3D.mesh = mesh
/// ```
#[derive(GodotClass)]
#[class(base=RefCounted, init)]
pub struct VoxelMesher {
    /// Color mapping for voxel types (voxel_value -> Color)
    colors: std::collections::HashMap<u16, Color>,
    /// Size of each voxel in world units
    voxel_size: f32,
    /// Whether to use greedy meshing optimization
    greedy_meshing: bool,
    base: Base<RefCounted>,
}

#[godot_api]
impl VoxelMesher {
    /// Creates a new VoxelMesher with default settings.
    #[func]
    pub fn create() -> Gd<Self> {
        let mut colors = std::collections::HashMap::new();
        // Default color palette
        colors.insert(1, Color::from_rgb(0.5, 0.4, 0.3)); // Brown (dirt/stone)
        colors.insert(2, Color::from_rgb(0.3, 0.7, 0.3)); // Green (grass)
        colors.insert(3, Color::from_rgb(0.3, 0.5, 0.9)); // Blue (water)
        colors.insert(4, Color::from_rgb(0.8, 0.8, 0.2)); // Yellow (sand)
        colors.insert(5, Color::from_rgb(0.6, 0.6, 0.6)); // Gray (stone)

        Gd::from_init_fn(|base| Self {
            colors,
            voxel_size: 1.0,
            greedy_meshing: false,
            base,
        })
    }

    /// Sets the color for a specific voxel type.
    ///
    /// # Arguments
    /// * `voxel_type` - The voxel value (1-65535)
    /// * `color` - The color to use for this voxel type
    #[func]
    pub fn set_voxel_color(&mut self, voxel_type: i32, color: Color) {
        if voxel_type > 0 && voxel_type <= 65535 {
            self.colors.insert(voxel_type as u16, color);
        }
    }

    /// Gets the color for a specific voxel type.
    #[func]
    pub fn get_voxel_color(&self, voxel_type: i32) -> Color {
        if voxel_type > 0 && voxel_type <= 65535 {
            *self
                .colors
                .get(&(voxel_type as u16))
                .unwrap_or(&Color::from_rgb(1.0, 0.0, 1.0)) // Magenta for unknown
        } else {
            Color::from_rgb(1.0, 0.0, 1.0)
        }
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

    /// Enables or disables greedy meshing optimization.
    ///
    /// Greedy meshing combines adjacent faces with the same voxel type into
    /// larger quads, significantly reducing polygon count.
    #[func]
    pub fn set_greedy_meshing(&mut self, enabled: bool) {
        self.greedy_meshing = enabled;
    }

    /// Returns whether greedy meshing is enabled.
    #[func]
    pub fn is_greedy_meshing_enabled(&self) -> bool {
        self.greedy_meshing
    }

    /// Generates a mesh from a VoxelTree.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `tree` - The VoxelTree to generate a mesh from
    ///
    /// # Returns
    /// An ArrayMesh containing the rendered voxels
    #[func]
    pub fn generate_mesh(&self, interner: Gd<VoxelInterner>, tree: Gd<VoxelTree>) -> Gd<ArrayMesh> {
        let tree_bind = tree.bind();
        let Some(ref vox_tree) = tree_bind.tree else {
            godot_error!("VoxelTree not initialized");
            return ArrayMesh::new_gd();
        };

        let interner_arc = interner.bind().get_arc();
        let interner_guard = interner_arc.read().unwrap();

        let max_depth = tree_bind.max_depth;
        let resolution = 1i32 << max_depth;

        if self.greedy_meshing {
            self.generate_greedy_mesh_internal(&interner_guard, vox_tree, resolution)
        } else {
            self.generate_simple_mesh_internal(&interner_guard, vox_tree, resolution)
        }
    }

    /// Generates a mesh from a specific chunk in a VoxelWorld.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `world` - The VoxelWorld containing the chunk
    /// * `chunk_pos` - The chunk position to mesh
    ///
    /// # Returns
    /// An ArrayMesh containing the rendered chunk, positioned at world coordinates
    #[func]
    pub fn generate_chunk_mesh(
        &self,
        interner: Gd<VoxelInterner>,
        world: Gd<VoxelWorld>,
        chunk_pos: Vector3i,
    ) -> Gd<ArrayMesh> {
        let world_bind = world.bind();
        let key = (chunk_pos.x, chunk_pos.y, chunk_pos.z);

        let Some(chunk) = world_bind.chunks.get(&key) else {
            return ArrayMesh::new_gd(); // Empty chunk
        };

        let chunk_bind = chunk.bind();
        let Some(ref vox_tree) = chunk_bind.tree else {
            return ArrayMesh::new_gd();
        };

        let interner_arc = interner.bind().get_arc();
        let interner_guard = interner_arc.read().unwrap();

        let resolution = 1i32 << chunk_bind.max_depth;
        let chunk_offset = IVec3::new(
            chunk_pos.x * resolution,
            chunk_pos.y * resolution,
            chunk_pos.z * resolution,
        );

        if self.greedy_meshing {
            self.generate_greedy_mesh_with_offset(&interner_guard, vox_tree, resolution, chunk_offset)
        } else {
            self.generate_simple_mesh_with_offset(&interner_guard, vox_tree, resolution, chunk_offset)
        }
    }

    /// Internal method to generate a simple culled-face mesh
    fn generate_simple_mesh_internal(
        &self,
        interner: &VoxInterner<u16>,
        tree: &VoxTree<u16>,
        resolution: i32,
    ) -> Gd<ArrayMesh> {
        self.generate_simple_mesh_with_offset(interner, tree, resolution, IVec3::ZERO)
    }

    /// Internal method to generate a simple culled-face mesh with position offset
    fn generate_simple_mesh_with_offset(
        &self,
        interner: &VoxInterner<u16>,
        tree: &VoxTree<u16>,
        resolution: i32,
        offset: IVec3,
    ) -> Gd<ArrayMesh> {
        let mut vertices = PackedVector3Array::new();
        let mut normals = PackedVector3Array::new();
        let mut colors = PackedColorArray::new();
        let mut uvs = PackedVector2Array::new();
        let mut indices = PackedInt32Array::new();

        let mut vertex_count = 0i32;

        // Iterate through all voxels
        for x in 0..resolution {
            for y in 0..resolution {
                for z in 0..resolution {
                    let pos = IVec3::new(x, y, z);
                    let voxel = tree.get(interner, pos);

                    if let Some(voxel_type) = voxel {
                        if voxel_type == 0 {
                            continue; // Empty voxel
                        }

                        let color = self
                            .colors
                            .get(&voxel_type)
                            .copied()
                            .unwrap_or(Color::from_rgb(1.0, 0.0, 1.0));

                        // Check each face direction
                        for dir in ALL_DIRECTIONS {
                            let neighbor_pos = pos + dir.offset();

                            // Check if neighbor is empty or out of bounds
                            let is_face_visible = if neighbor_pos.x < 0
                                || neighbor_pos.x >= resolution
                                || neighbor_pos.y < 0
                                || neighbor_pos.y >= resolution
                                || neighbor_pos.z < 0
                                || neighbor_pos.z >= resolution
                            {
                                true // Out of bounds = visible
                            } else {
                                let neighbor = tree.get(interner, neighbor_pos);
                                neighbor.is_none() || neighbor == Some(0)
                            };

                            if is_face_visible {
                                // Add face vertices (with offset applied)
                                let world_pos = pos + offset;
                                let face_verts = dir.vertices(world_pos, self.voxel_size);
                                let face_uvs = dir.uvs();
                                let normal = dir.normal();

                                for (vert, uv) in face_verts.iter().zip(face_uvs.iter()) {
                                    vertices.push(*vert);
                                    normals.push(normal);
                                    colors.push(color);
                                    uvs.push(*uv);
                                }

                                // Add indices for two triangles (counter-clockwise)
                                indices.push(vertex_count);
                                indices.push(vertex_count + 1);
                                indices.push(vertex_count + 2);
                                indices.push(vertex_count);
                                indices.push(vertex_count + 2);
                                indices.push(vertex_count + 3);

                                vertex_count += 4;
                            }
                        }
                    }
                }
            }
        }

        self.build_array_mesh(vertices, normals, colors, uvs, indices)
    }

    /// Internal method to generate a greedy-meshed mesh
    fn generate_greedy_mesh_internal(
        &self,
        interner: &VoxInterner<u16>,
        tree: &VoxTree<u16>,
        resolution: i32,
    ) -> Gd<ArrayMesh> {
        self.generate_greedy_mesh_with_offset(interner, tree, resolution, IVec3::ZERO)
    }

    /// Internal method to generate a greedy-meshed mesh with position offset
    fn generate_greedy_mesh_with_offset(
        &self,
        interner: &VoxInterner<u16>,
        tree: &VoxTree<u16>,
        resolution: i32,
        offset: IVec3,
    ) -> Gd<ArrayMesh> {
        let mut vertices = PackedVector3Array::new();
        let mut normals = PackedVector3Array::new();
        let mut colors = PackedColorArray::new();
        let mut uvs = PackedVector2Array::new();
        let mut indices = PackedInt32Array::new();

        let mut vertex_count = 0i32;
        let res = resolution as usize;

        // Process each axis direction
        for dir in ALL_DIRECTIONS {
            // Create a mask for tracking which faces have been processed
            let mut processed = vec![vec![vec![false; res]; res]; res];

            // Iterate based on the primary axis of this direction
            match dir {
                FaceDirection::Top | FaceDirection::Bottom => {
                    for y in 0..resolution {
                        for x in 0..resolution {
                            for z in 0..resolution {
                                self.try_greedy_expand(
                                    interner,
                                    tree,
                                    resolution,
                                    offset,
                                    IVec3::new(x, y, z),
                                    dir,
                                    &mut processed,
                                    &mut vertices,
                                    &mut normals,
                                    &mut colors,
                                    &mut uvs,
                                    &mut indices,
                                    &mut vertex_count,
                                );
                            }
                        }
                    }
                }
                FaceDirection::East | FaceDirection::West => {
                    for x in 0..resolution {
                        for y in 0..resolution {
                            for z in 0..resolution {
                                self.try_greedy_expand(
                                    interner,
                                    tree,
                                    resolution,
                                    offset,
                                    IVec3::new(x, y, z),
                                    dir,
                                    &mut processed,
                                    &mut vertices,
                                    &mut normals,
                                    &mut colors,
                                    &mut uvs,
                                    &mut indices,
                                    &mut vertex_count,
                                );
                            }
                        }
                    }
                }
                FaceDirection::North | FaceDirection::South => {
                    for z in 0..resolution {
                        for x in 0..resolution {
                            for y in 0..resolution {
                                self.try_greedy_expand(
                                    interner,
                                    tree,
                                    resolution,
                                    offset,
                                    IVec3::new(x, y, z),
                                    dir,
                                    &mut processed,
                                    &mut vertices,
                                    &mut normals,
                                    &mut colors,
                                    &mut uvs,
                                    &mut indices,
                                    &mut vertex_count,
                                );
                            }
                        }
                    }
                }
            }
        }

        self.build_array_mesh(vertices, normals, colors, uvs, indices)
    }

    /// Try to expand a greedy mesh quad from the given starting position
    #[allow(clippy::too_many_arguments)]
    fn try_greedy_expand(
        &self,
        interner: &VoxInterner<u16>,
        tree: &VoxTree<u16>,
        resolution: i32,
        offset: IVec3,
        start_pos: IVec3,
        dir: FaceDirection,
        processed: &mut [Vec<Vec<bool>>],
        vertices: &mut PackedVector3Array,
        normals: &mut PackedVector3Array,
        colors: &mut PackedColorArray,
        uvs: &mut PackedVector2Array,
        indices: &mut PackedInt32Array,
        vertex_count: &mut i32,
    ) {
        let (x, y, z) = (
            start_pos.x as usize,
            start_pos.y as usize,
            start_pos.z as usize,
        );

        // Skip if already processed
        if processed[x][y][z] {
            return;
        }

        // Check if this voxel has a visible face in this direction
        let voxel = tree.get(interner, start_pos);
        let voxel_type = match voxel {
            Some(v) if v > 0 => v,
            _ => return,
        };

        let neighbor_pos = start_pos + dir.offset();
        let is_face_visible = if neighbor_pos.x < 0
            || neighbor_pos.x >= resolution
            || neighbor_pos.y < 0
            || neighbor_pos.y >= resolution
            || neighbor_pos.z < 0
            || neighbor_pos.z >= resolution
        {
            true
        } else {
            let neighbor = tree.get(interner, neighbor_pos);
            neighbor.is_none() || neighbor == Some(0)
        };

        if !is_face_visible {
            processed[x][y][z] = true;
            return;
        }

        // Determine expansion axes based on face direction
        let (axis1, axis2) = match dir {
            FaceDirection::Top | FaceDirection::Bottom => (IVec3::X, IVec3::Z),
            FaceDirection::East | FaceDirection::West => (IVec3::Z, IVec3::Y),
            FaceDirection::North | FaceDirection::South => (IVec3::X, IVec3::Y),
        };

        // Expand along axis1
        let mut width = 1i32;
        loop {
            let check_pos = start_pos + axis1 * width;
            if check_pos.x >= resolution || check_pos.y >= resolution || check_pos.z >= resolution {
                break;
            }
            if check_pos.x < 0 || check_pos.y < 0 || check_pos.z < 0 {
                break;
            }

            let (cx, cy, cz) = (
                check_pos.x as usize,
                check_pos.y as usize,
                check_pos.z as usize,
            );
            if processed[cx][cy][cz] {
                break;
            }

            let check_voxel = tree.get(interner, check_pos);
            if check_voxel != Some(voxel_type) {
                break;
            }

            let check_neighbor = check_pos + dir.offset();
            let check_visible = if check_neighbor.x < 0
                || check_neighbor.x >= resolution
                || check_neighbor.y < 0
                || check_neighbor.y >= resolution
                || check_neighbor.z < 0
                || check_neighbor.z >= resolution
            {
                true
            } else {
                let n = tree.get(interner, check_neighbor);
                n.is_none() || n == Some(0)
            };

            if !check_visible {
                break;
            }

            width += 1;
        }

        // Expand along axis2
        let mut height = 1i32;
        'outer: loop {
            let row_start = start_pos + axis2 * height;
            if row_start.x >= resolution || row_start.y >= resolution || row_start.z >= resolution {
                break;
            }
            if row_start.x < 0 || row_start.y < 0 || row_start.z < 0 {
                break;
            }

            for w in 0..width {
                let check_pos = row_start + axis1 * w;
                if check_pos.x >= resolution
                    || check_pos.y >= resolution
                    || check_pos.z >= resolution
                {
                    break 'outer;
                }
                if check_pos.x < 0 || check_pos.y < 0 || check_pos.z < 0 {
                    break 'outer;
                }

                let (cx, cy, cz) = (
                    check_pos.x as usize,
                    check_pos.y as usize,
                    check_pos.z as usize,
                );
                if processed[cx][cy][cz] {
                    break 'outer;
                }

                let check_voxel = tree.get(interner, check_pos);
                if check_voxel != Some(voxel_type) {
                    break 'outer;
                }

                let check_neighbor = check_pos + dir.offset();
                let check_visible = if check_neighbor.x < 0
                    || check_neighbor.x >= resolution
                    || check_neighbor.y < 0
                    || check_neighbor.y >= resolution
                    || check_neighbor.z < 0
                    || check_neighbor.z >= resolution
                {
                    true
                } else {
                    let n = tree.get(interner, check_neighbor);
                    n.is_none() || n == Some(0)
                };

                if !check_visible {
                    break 'outer;
                }
            }

            height += 1;
        }

        // Mark all covered voxels as processed
        for h in 0..height {
            for w in 0..width {
                let mark_pos = start_pos + axis1 * w + axis2 * h;
                let (mx, my, mz) = (
                    mark_pos.x as usize,
                    mark_pos.y as usize,
                    mark_pos.z as usize,
                );
                processed[mx][my][mz] = true;
            }
        }

        // Generate the merged quad
        let color = self
            .colors
            .get(&voxel_type)
            .copied()
            .unwrap_or(Color::from_rgb(1.0, 0.0, 1.0));

        let world_pos = start_pos + offset;
        let end_pos = world_pos + axis1 * width + axis2 * height;

        // Generate vertices for the merged quad
        let face_verts = self.greedy_face_vertices(world_pos, end_pos, dir);
        let normal = dir.normal();

        let base_uvs = FaceDirection::Top.uvs();
        for (i, vert) in face_verts.iter().enumerate() {
            vertices.push(*vert);
            normals.push(normal);
            colors.push(color);
            // Scale UVs based on quad size for tiling
            let base_uv = base_uvs[i];
            uvs.push(Vector2::new(base_uv.x * width as f32, base_uv.y * height as f32));
        }

        // Add indices
        indices.push(*vertex_count);
        indices.push(*vertex_count + 1);
        indices.push(*vertex_count + 2);
        indices.push(*vertex_count);
        indices.push(*vertex_count + 2);
        indices.push(*vertex_count + 3);

        *vertex_count += 4;
    }

    /// Generate vertices for a greedy-merged face quad
    fn greedy_face_vertices(
        &self,
        start: IVec3,
        end: IVec3,
        dir: FaceDirection,
    ) -> [Vector3; 4] {
        let s = self.voxel_size;
        let p0 = Vector3::new(start.x as f32 * s, start.y as f32 * s, start.z as f32 * s);
        let p1 = Vector3::new(end.x as f32 * s, end.y as f32 * s, end.z as f32 * s);

        match dir {
            FaceDirection::Top => [
                Vector3::new(p0.x, p1.y, p0.z),
                Vector3::new(p0.x, p1.y, p1.z),
                Vector3::new(p1.x, p1.y, p1.z),
                Vector3::new(p1.x, p1.y, p0.z),
            ],
            FaceDirection::Bottom => [
                Vector3::new(p0.x, p0.y, p1.z),
                Vector3::new(p0.x, p0.y, p0.z),
                Vector3::new(p1.x, p0.y, p0.z),
                Vector3::new(p1.x, p0.y, p1.z),
            ],
            FaceDirection::North => [
                Vector3::new(p1.x, p0.y, p1.z),
                Vector3::new(p1.x, p1.y, p1.z),
                Vector3::new(p0.x, p1.y, p1.z),
                Vector3::new(p0.x, p0.y, p1.z),
            ],
            FaceDirection::South => [
                Vector3::new(p0.x, p0.y, p0.z),
                Vector3::new(p0.x, p1.y, p0.z),
                Vector3::new(p1.x, p1.y, p0.z),
                Vector3::new(p1.x, p0.y, p0.z),
            ],
            FaceDirection::East => [
                Vector3::new(p1.x, p0.y, p0.z),
                Vector3::new(p1.x, p1.y, p0.z),
                Vector3::new(p1.x, p1.y, p1.z),
                Vector3::new(p1.x, p0.y, p1.z),
            ],
            FaceDirection::West => [
                Vector3::new(p0.x, p0.y, p1.z),
                Vector3::new(p0.x, p1.y, p1.z),
                Vector3::new(p0.x, p1.y, p0.z),
                Vector3::new(p0.x, p0.y, p0.z),
            ],
        }
    }

    /// Build the final ArrayMesh from collected data
    fn build_array_mesh(
        &self,
        vertices: PackedVector3Array,
        normals: PackedVector3Array,
        colors: PackedColorArray,
        uvs: PackedVector2Array,
        indices: PackedInt32Array,
    ) -> Gd<ArrayMesh> {
        let mut mesh = ArrayMesh::new_gd();

        if vertices.is_empty() {
            return mesh;
        }

        // ArrayType::MAX is 13, so we need 13 elements in the array
        let mut arrays = VarArray::new();
        arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());

        arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
        arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
        arrays.set(ArrayType::COLOR.ord() as usize, &colors.to_variant());
        arrays.set(ArrayType::TEX_UV.ord() as usize, &uvs.to_variant());
        arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

        mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);

        mesh
    }

    /// Returns mesh statistics for debugging.
    ///
    /// # Arguments
    /// * `interner` - The VoxelInterner managing memory
    /// * `tree` - The VoxelTree to analyze
    ///
    /// # Returns
    /// A dictionary with vertex_count, face_count, and triangle_count
    #[func]
    pub fn get_mesh_stats(&self, interner: Gd<VoxelInterner>, tree: Gd<VoxelTree>) -> VarDictionary {
        let mesh = self.generate_mesh(interner, tree);
        let mut dict = VarDictionary::new();

        if mesh.get_surface_count() > 0 {
            let arrays = mesh.surface_get_arrays(0);
            let vertices: PackedVector3Array = arrays
                .get(ArrayType::VERTEX.ord() as usize)
                .map(|v| v.try_to().unwrap_or_default())
                .unwrap_or_default();
            let indices: PackedInt32Array = arrays
                .get(ArrayType::INDEX.ord() as usize)
                .map(|v| v.try_to().unwrap_or_default())
                .unwrap_or_default();

            dict.set("vertex_count", vertices.len() as i32);
            dict.set("face_count", vertices.len() as i32 / 4);
            dict.set("triangle_count", indices.len() as i32 / 3);
        } else {
            dict.set("vertex_count", 0);
            dict.set("face_count", 0);
            dict.set("triangle_count", 0);
        }

        dict
    }
}
