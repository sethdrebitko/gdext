/*
 * Copyright (c) godot-rust; Bromeon and contributors.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use godot::prelude::*;
use glam_029::IVec3;
use voxelis::interner::MAX_ALLOWED_DEPTH;
use voxelis::spatial::{VoxOpsBulkWrite, VoxOpsRead, VoxOpsWrite, VoxTree};
use voxelis::{MaxDepth, VoxInterner};

const DEFAULT_MAX_DEPTH: u8 = 5;
const DEFAULT_MEMORY_BUDGET_MB: u64 = 256;
const BYTES_PER_MB: u64 = 1024 * 1024;

#[derive(GodotClass)]
#[class(init, base=Node)]
pub struct VoxelWorld {
    base: Base<Node>,
    max_depth: u8,
    memory_budget_mb: u64,
    interner: VoxInterner<u8>,
    tree: VoxTree<u8>,
}

#[godot_api]
impl VoxelWorld {
    fn init(base: Base<Node>) -> Self {
        let max_depth = DEFAULT_MAX_DEPTH;
        let memory_budget_mb = DEFAULT_MEMORY_BUDGET_MB;
        let (interner, tree) = Self::build_storage(max_depth, memory_budget_mb);

        Self {
            base,
            max_depth,
            memory_budget_mb,
            interner,
            tree,
        }
    }

    #[func]
    fn configure(&mut self, max_depth: i32, memory_budget_mb: i64) -> bool {
        let Some(max_depth) = Self::validate_max_depth(max_depth) else {
            let max_allowed = MAX_ALLOWED_DEPTH.saturating_sub(1);
            godot_warn!(
                "VoxelWorld.configure: max_depth must be 0..={max_allowed}, got {max_depth}"
            );
            return false;
        };

        let Some(memory_budget_mb) = Self::validate_memory_budget(memory_budget_mb) else {
            godot_warn!(
                "VoxelWorld.configure: memory_budget_mb must be > 0, got {memory_budget_mb}"
            );
            return false;
        };

        let (interner, tree) = Self::build_storage(max_depth, memory_budget_mb);

        self.max_depth = max_depth;
        self.memory_budget_mb = memory_budget_mb;
        self.interner = interner;
        self.tree = tree;

        true
    }

    #[func]
    fn set_voxel(&mut self, position: Vector3i, value: i32) -> bool {
        let Some(voxel) = Self::validate_voxel_value(value) else {
            godot_warn!("VoxelWorld.set_voxel: value must be 0..255, got {value}");
            return false;
        };

        let position = Self::to_ivec3(position);
        if !self.in_bounds(position) {
            godot_warn!("VoxelWorld.set_voxel: position out of bounds: {position:?}");
            return false;
        }

        self.tree.set(&mut self.interner, position, voxel)
    }

    #[func]
    fn get_voxel(&self, position: Vector3i) -> i32 {
        let position = Self::to_ivec3(position);
        if !self.in_bounds(position) {
            return -1;
        }

        self.tree
            .get(&self.interner, position)
            .map(|voxel| voxel as i32)
            .unwrap_or(0)
    }

    #[func]
    fn fill(&mut self, value: i32) -> bool {
        let Some(voxel) = Self::validate_voxel_value(value) else {
            godot_warn!("VoxelWorld.fill: value must be 0..255, got {value}");
            return false;
        };

        self.tree.fill(&mut self.interner, voxel);
        true
    }

    #[func]
    fn clear(&mut self) {
        self.tree.clear(&mut self.interner);
    }

    #[func]
    fn get_voxel_bounds(&self) -> Vector3i {
        let axis = self.axis_len();
        Vector3i::new(axis, axis, axis)
    }

    #[func]
    fn get_max_depth(&self) -> i32 {
        self.max_depth as i32
    }

    #[func]
    fn get_memory_budget_mb(&self) -> i64 {
        self.memory_budget_mb as i64
    }
}

impl VoxelWorld {
    fn build_storage(max_depth: u8, memory_budget_mb: u64) -> (VoxInterner<u8>, VoxTree<u8>) {
        let bytes = Self::memory_budget_bytes(memory_budget_mb);
        let interner = VoxInterner::with_memory_budget(bytes);
        let tree = VoxTree::new(MaxDepth::new(max_depth));
        (interner, tree)
    }

    fn memory_budget_bytes(memory_budget_mb: u64) -> usize {
        let bytes = memory_budget_mb.saturating_mul(BYTES_PER_MB);
        bytes.min(usize::MAX as u64) as usize
    }

    fn axis_len(&self) -> i32 {
        1_i32 << (self.max_depth as u32)
    }

    fn to_ivec3(position: Vector3i) -> IVec3 {
        IVec3::new(position.x, position.y, position.z)
    }

    fn in_bounds(&self, position: IVec3) -> bool {
        let axis = self.axis_len();
        position.x >= 0
            && position.y >= 0
            && position.z >= 0
            && position.x < axis
            && position.y < axis
            && position.z < axis
    }

    fn validate_max_depth(max_depth: i32) -> Option<u8> {
        let max_depth = u8::try_from(max_depth).ok()?;
        MaxDepth::try_from(max_depth).ok()?;
        Some(max_depth)
    }

    fn validate_memory_budget(memory_budget_mb: i64) -> Option<u64> {
        if memory_budget_mb <= 0 {
            return None;
        }

        Some(memory_budget_mb as u64)
    }

    fn validate_voxel_value(value: i32) -> Option<u8> {
        u8::try_from(value).ok()
    }
}
