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
//! ```

use std::sync::{Arc, RwLock};

use glam::IVec3;
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
