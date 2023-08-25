# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of NeuS.
Modified by Yuanhui Huang.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type
from typing_extensions import Literal

import torch

from nerfstudio.cameras.rays import RayBundle

# from nerfstudio.engine.callbacks import (
#     TrainingCallback,
#     TrainingCallbackAttributes,
#     TrainingCallbackLocation,
# )
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.fields.sdf_custom_field import SDFCustomFieldConfig

from mmengine import MMLogger


@dataclass
class NeuSCustomModelConfig(SurfaceModelConfig):
    """NeuS Model Config"""

    _target: Type = field(default_factory=lambda: NeuSCustomModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""

    sdf_field: SDFCustomFieldConfig = SDFCustomFieldConfig()
    """Config for SDF Field"""

    depth_method: Literal["median", "expected"] = "expected"

    beta_hand_tune: bool = False
    beta_min: float = 2.0
    beta_max: float = 12.0
    total_iters: int = 12 * 3516

    anneal_aabb: bool = False
    aabb_min_near: float = 10.0
    aabb_max_near: float = 0.2
    aabb_min_far_frac: float = 0.25
    aabb_every_iters: int = 3516
    aabb_original: List[float] = field(default_factory=lambda: [-81.0, -81.0, -4.0, 81.0, 81.0, 12.0])


class NeuSCustomModel(SurfaceModel):
    """NeuS model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuSCustomModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        print(
            f"NeuSCustomModel Config: num_samples {self.config.num_samples}, num_samples_importance {self.config.num_samples_importance}, num_up_sample_steps {self.config.num_up_sample_steps}"
        )

        self.sampler = NeuSSampler(
            num_samples=self.config.num_samples,
            num_samples_importance=self.config.num_samples_importance,
            # flag
            num_samples_outside=self.config.num_samples_outside,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
            # beta_hand_tune=self.config.beta_hand_tune,
            # beta_min=self.config.beta_min,
            # beta_max=self.config.beta_max
        )

        # flag
        self.anneal_end = 5000
        # self.beta = self.config.beta_min

        self.renderer_depth.method = self.config.depth_method
        aabb = self.config.aabb_original
        self.aabb = torch.tensor([aabb[:3], aabb[3:]])

    # def get_training_callbacks(
    #     self, training_callback_attributes: TrainingCallbackAttributes
    # ) -> List[TrainingCallback]:
    #     callbacks = super().get_training_callbacks(training_callback_attributes)
    #     # anneal for cos in NeuS
    #     if self.anneal_end > 0:

    #         def set_anneal(step):
    #             anneal = min([1.0, step / self.anneal_end])
    #             self.field.set_cos_anneal_ratio(anneal)

    #         callbacks.append(
    #             TrainingCallback(
    #                 where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
    #                 update_every_num_iters=1,
    #                 func=set_anneal,
    #             )
    #         )

    #     return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        # if self.config.beta_hand_tune:
        #     beta = self.beta
        # else:
        #     beta = None
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        # save_points("a.ply", ray_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict

    def forward(self, ray_bundle: RayBundle, iter: int = None) -> Dict[str, torch.Tensor]:
        if self.anneal_end > 0 and self.training and iter is not None:

            def set_anneal(step):
                anneal = min([1.0, step / self.anneal_end])
                self.field.set_cos_anneal_ratio(anneal)

            set_anneal(iter)

        if self.config.beta_hand_tune and self.training and iter is not None:
            beta = min(
                self.config.beta_max,
                self.config.beta_min + (self.config.beta_max - self.config.beta_min) * iter / self.config.total_iters,
            )
            self.field.deviation_network.variance.data = beta * torch.ones(1, device=ray_bundle.origins.device)

        if self.config.anneal_aabb and iter is not None and iter % self.config.aabb_every_iters == 0:
            near = max(
                self.config.aabb_max_near,
                self.config.aabb_min_near
                - iter / self.config.total_iters * (self.config.aabb_min_near - self.config.aabb_max_near),
            )
            aabb_coef = min(
                1.0,
                self.config.aabb_min_far_frac + iter / self.config.total_iters * (1 - self.config.aabb_min_far_frac),
            )
            self.collider.near_plane = near
            self.collider.scene_box.aabb[:, :2] = aabb_coef * self.aabb[:, :2]
            logger = MMLogger.get_instance("selfocc")
            logger.info(f"aabb_annealed! near: {self.collider.near_plane}, aabb: {self.collider.scene_box.aabb}")

        return super().forward(ray_bundle)
