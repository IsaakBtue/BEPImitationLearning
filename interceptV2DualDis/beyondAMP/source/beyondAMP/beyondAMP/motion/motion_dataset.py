from __future__ import annotations

import os
import numpy as np
import torch
from typing import Sequence, List, Union
from dataclasses import MISSING

try:
    from isaaclab.utils import configclass
except ImportError:
    # Fallback for backends (e.g. mjlab) that don't depend on IsaacLab.
    # IsaacLab's configclass lets fields default to the MISSING sentinel and
    # still come after fields with concrete defaults; stdlib dataclass forbids
    # that. We rewrite ``= MISSING`` as ``field(default_factory=lambda: MISSING)``
    # so dataclass sees a default (and ``field(default=MISSING)`` itself means
    # "no default", so we have to use the factory route).
    from dataclasses import dataclass as _dataclass, field as _field

    def _missing_factory():
        return MISSING

    def configclass(cls):  # type: ignore[no-redef]
        for name, value in list(cls.__dict__.items()):
            if value is MISSING:
                setattr(cls, name, _field(default_factory=_missing_factory))
        return _dataclass(cls)

from .utils.math import quat_apply_inverse, quat_conjugate, quat_apply
from .motion_transition import MotionTransition

class MotionDataset:
    """
    Load multiple motion files and build (s_t, s_{t+1}) transitions.
    Efficient contiguous tensors + pre-built index mapping.
    """

    def __init__(
        self, 
        cfg: MotionDatasetCfg,
        env,
        device: str = "cpu",
        ):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.robot = env.scene[cfg.asset_name]
        self.motion_files = cfg.motion_files
        self.observation_terms = cfg.amp_obs_terms
        
        body_names = cfg.body_names
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(body_names, preserve_order=True)[0], dtype=torch.long, device=device
        )

        # FIX 2026-07-15: optional joint subset for joint_pos/joint_vel (see
        # MotionDatasetCfg.joint_names). None -> all joints, matching prior
        # (only) behavior exactly.
        if cfg.joint_names is not None:
            self.joint_indexes = torch.tensor(
                self.robot.find_joints(cfg.joint_names, preserve_order=True)[0], dtype=torch.long, device=device
            )
        else:
            self.joint_indexes = None

        # FIX 2026-07-16: optional joint MASK (as opposed to joint_names'
        # SLICE) -- freezes the given joints' columns to the robot's default
        # pose/zero-velocity across the whole dataset, keeping the tensor's
        # full joint-count shape (needed when different regions' datasets
        # must stay the same dimension, e.g. this project's multi-disc AMP
        # routing one shared amp_obs tensor to per-region discriminators).
        # Applied in load_motions() below, once robot.data.default_joint_pos
        # is available.
        if cfg.freeze_joint_names is not None:
            self.freeze_joint_indexes = torch.tensor(
                self.robot.find_joints(cfg.freeze_joint_names, preserve_order=True)[0],
                dtype=torch.long, device=device,
            )
        else:
            self.freeze_joint_indexes = None

        anchor_name = cfg.anchor_name
        self.anchor_index = torch.tensor(
            self.robot.find_bodies(anchor_name, preserve_order=True)[0], dtype=torch.long, device=device
        )

        self.load_motions()
        self.init_observation_dims()

    def load_motions(self):
        # Storage lists (later concatenated)
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        fps_list = []
        traj_lengths = []

        # Load all motion files
        for f in self.motion_files:
            assert os.path.isfile(f), f"Invalid motion file: {f}"
            data = np.load(f)

            fps_list.append(float(data["fps"]))
            traj_len = data["joint_pos"].shape[0]
            traj_lengths.append(traj_len)

            joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
            joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32))
            body_pos_w_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32))
            body_quat_w_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32))
            body_lin_vel_w_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32))
            body_ang_vel_w_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32))

        # Concatenate all trajectories into single big tensors
        self.joint_pos_all  = torch.cat(joint_pos_list, dim=0).to(self.device)
        self.joint_vel_all  = torch.cat(joint_vel_list, dim=0).to(self.device)

        # FIX 2026-07-16: freeze masked joints (see MotionDatasetCfg.
        # freeze_joint_names) to the robot's default pose / zero velocity
        # across every frame of this dataset -- makes those columns
        # uninformative to a discriminator trained on this data, without
        # changing the tensor's joint-count shape.
        if self.freeze_joint_indexes is not None:
            default_pos = self.robot.data.default_joint_pos[0, self.freeze_joint_indexes].to(self.device)
            self.joint_pos_all[:, self.freeze_joint_indexes] = default_pos
            self.joint_vel_all[:, self.freeze_joint_indexes] = 0.0

        self.body_pos_w_all      = torch.cat(body_pos_w_list, dim=0).to(self.device)
        self.body_quat_w_all     = torch.cat(body_quat_w_list, dim=0).to(self.device)
        self.body_lin_vel_w_all  = torch.cat(body_lin_vel_w_list, dim=0).to(self.device)
        self.body_ang_vel_w_all  = torch.cat(body_ang_vel_w_list, dim=0).to(self.device)

        self.total_dataset_size = sum(traj_lengths)

        # Keep per-trajectory FPS if needed
        self.fps_list = fps_list
        self._traj_lengths = traj_lengths

        # Build transition index list: (global_index_t, global_index_t+1)
        self.index_t, self.index_tp1 = self._build_transition_indices(traj_lengths, self.device)

    # ----------------------- Property API -----------------------

    def subtract_flaten(self, target: torch.Tensor):
        target = target[:, self.body_indexes]
        return target.reshape(self.total_dataset_size, -1)

    @property
    def joint_pos(self):
        if self.joint_indexes is None:
            return self.joint_pos_all
        return self.joint_pos_all[:, self.joint_indexes]

    @property
    def joint_vel(self):
        if self.joint_indexes is None:
            return self.joint_vel_all
        return self.joint_vel_all[:, self.joint_indexes]

    @property
    def body_pos_w(self):
        return self.body_pos_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_quat_w(self):
        return self.body_quat_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    @property
    def body_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.body_indexes].reshape(self.total_dataset_size, -1)
    
    @property
    def body_pos_b(self):
        """
        body positions expressed in anchor-local frame.
        Output: (N, num_bodies * 3)
        """
        # (N, B, 3)
        pos_w = self.body_pos_w_all[:, self.body_indexes]  

        # (N, 1, 3)
        anchor_pos = self._anchor_pos.unsqueeze(1)
        anchor_quat = self._anchor_quat.unsqueeze(1)

        # translate then rotate into anchor frame
        rel = pos_w - anchor_pos                           # world-space relative
        rel_local = quat_apply_inverse(anchor_quat, rel)   # world → anchor

        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_quat_b(self):
        """
        body orientations expressed in anchor-local frame.
        q_local = q_anchor^{-1} ⊗ q_body
        Output: (N, num_bodies * 4)
        """
        q_body = self.body_quat_w_all[:, self.body_indexes]             # (N, B, 4)
        q_anchor = self._anchor_quat.unsqueeze(1)                       # (N, 1, 4)

        q_anchor_inv = quat_conjugate(q_anchor)                         # IsaacLab: unit quats → inverse = conjugate
        q_rel = quat_apply(q_anchor_inv, q_body)                        # broadcast quaternion multiply

        return q_rel.reshape(self.total_dataset_size, -1)

    @property
    def body_lin_vel_b(self):
        """
        body linear velocities in anchor-local frame.
        v_rel_local = R(q_anchor)^T (v_body - v_anchor)
        """
        v_body = self.body_lin_vel_w_all[:, self.body_indexes]          # (N, B, 3)
        v_anchor = self.anchor_lin_vel_w.unsqueeze(1)                   # (N, 1, 3)

        rel = v_body - v_anchor                                         # world frame
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)

        return rel_local.reshape(self.total_dataset_size, -1)

    @property
    def body_ang_vel_b(self):
        """
        body angular velocities in anchor-local frame.
        ω_rel_local = R(q_anchor)^T (ω_body - ω_anchor)
        """
        w_body = self.body_ang_vel_w_all[:, self.body_indexes]          # (N, B, 3)
        w_anchor = self.anchor_ang_vel_w.unsqueeze(1)                   # (N, 1, 3)

        rel = w_body - w_anchor
        rel_local = quat_apply_inverse(self._anchor_quat.unsqueeze(1), rel)

        return rel_local.reshape(self.total_dataset_size, -1)

    
    @property
    def anchor_height(self):
        return self.anchor_pos_w[:, -1]
    
    @property
    def anchor_pos_w(self):
        return self.body_pos_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_quat_w(self):
        return self.body_quat_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w_all[:, self.anchor_index].reshape(self.total_dataset_size, -1)
        
    @property
    def base_lin_vel(self):
        """
        Base (anchor) linear velocity expressed in base frame.
        Shape: (N, 3)
        """
        v_w = self.anchor_lin_vel_w                       # (N, 3)
        q_w = self.anchor_quat_w                          # (N, 4)

        v_b = quat_apply_inverse(q_w, v_w)                # world → base
        return v_b

    @property
    def base_ang_vel(self):
        """
        Base (anchor) angular velocity expressed in base frame.
        Shape: (N, 3)
        """
        w_w = self.anchor_ang_vel_w                       # (N, 3)
        q_w = self.anchor_quat_w                          # (N, 4)

        w_b = quat_apply_inverse(q_w, w_w)                # world → base
        return w_b


    # ----------------------- Transition index builder -----------------------

    def observation_dim_cast(self, name)->int:
        # shape_cast_table = {
        #     "displacement": self.body_indexes.shape[-1]
        # }
        if hasattr(self, name):
            obs_term: torch.Tensor = getattr(self, name)
            assert isinstance(obs_term, torch.Tensor), f"invalid observation name: {name} for get dim"
            return obs_term.shape[-1]
        else:
            raise NotImplementedError(f"Failed for term: {name}")

    def init_observation_dims(self):
        observation_dims = []
        for obs_term in self.observation_terms:
            # observation_terms.append(obs_term)
            observation_dims.append(self.observation_dim_cast(obs_term))
        self.observation_dim = sum(observation_dims)
        self.observation_dims = observation_dims

    # ----------------------- Transition index builder -----------------------

    def _build_transition_indices(self, traj_lengths: List[int], device: str):
        """
        Build valid (t, t+1) pairs without crossing trajectory boundaries.
        """
        idx_t = []
        idx_tp1 = []

        offset = 0
        for L in traj_lengths:
            if L < 2:
                offset += L
                continue
            t = torch.arange(offset, offset + L - 1)
            idx_t.append(t)
            idx_tp1.append(t + 1)
            offset += L

        idx_t = torch.cat(idx_t).to(device)
        idx_tp1 = torch.cat(idx_tp1).to(device)
        return idx_t, idx_tp1

    # ----------------------- Batch Sampling API -----------------------

    def sample_batch(self, batch_size: int):
        """
        Sample a batch of transitions:
            s_t → s_{t+1}

        Returns dict:
            {
                "joint_pos_t": ...,
                "joint_pos_tp1": ...,
                ...
            }
        """
        idx = torch.randint(0, len(self.index_t), (batch_size,), device=self.device)
        t = self.index_t[idx]
        tp1 = self.index_tp1[idx]
        return t, tp1

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for idx in range(0, num_mini_batch):
            t, tp1 = self.sample_batch(mini_batch_size)
            res_t, res_tp1 = self.build_transition(t, tp1)
            yield res_t, res_tp1
            
    def build_transition(self, t, tp1):
        res_t, res_tp1 = [], []
        for term in self.observation_terms:
            _t, _tp1 = getattr(self, term)[t], getattr(self, term)[tp1]
            res_t.append(_t); res_tp1.append(_tp1)
        res_t, res_tp1 = torch.cat(res_t, dim=-1), torch.cat(res_tp1, dim=-1)
        return res_t, res_tp1
        

@configclass
class MotionDatasetCfg:
    class_type          : type[MotionDataset] = MotionDataset
    asset_name          : str = "robot"
    motion_files        : List[str] = MISSING
    body_names          : List[str] = MISSING
    amp_obs_terms       : List[str] = MISSING
    anchor_name         : str = MISSING
    motion_weights      : Union[List[float], None] = None
    # FIX 2026-07-15: optional joint subset for the "joint_pos"/"joint_vel"
    # observation terms (e.g. excluding arm joints from an AMP discriminator
    # that has no task-grounding reward for them). None = all joints
    # (previous, only) behavior.
    joint_names         : Union[List[str], None] = None
    # FIX 2026-07-16: optional joint MASK (as opposed to joint_names' SLICE)
    # -- freezes the given joints to the robot's default pose/zero velocity
    # across the whole dataset instead of removing them from the tensor
    # shape. Use when different regions/discriminators must keep the same
    # observation dimension but only some of them should see real motion
    # for these joints (e.g. arms kept live for "near" regions, frozen/
    # uninformative for "far" regions in the same multi-disc setup).
    freeze_joint_names  : Union[List[str], None] = None
