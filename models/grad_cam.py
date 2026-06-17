"""
Grad-CAM for multimodal satellite damage models.

Works with any model that has a CNNBackbone visual_backbone
(BaselineMultimodalModel, ImageDifferenceWeatherGRUModel,
ImageDifferenceCNNModel).  WeatherGRUModel has no visual branch
and is not supported.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Target-layer helper
# ──────────────────────────────────────────────────────────────────────────────

def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the last spatial feature block of the shared CNNBackbone.

    CNNBackbone.features layout:
      [0] DamageConvBlock(3 → 64)
      [1] MaxPool2d
      [2] DamageConvBlock(64 → 128)
      [3] MaxPool2d
      [4] DamageConvBlock(128 → out_dim)   ← last spatial block (H×W still > 1)
      [5] AdaptiveAvgPool2d(1)
    """
    if not hasattr(model, 'visual_backbone'):
        raise ValueError(
            f"{type(model).__name__} has no visual_backbone — "
            "Grad-CAM requires a CNNBackbone."
        )
    return model.visual_backbone.features[4]


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM engine
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalGradCAM:
    """
    Grad-CAM for models that call a shared CNNBackbone on pre then post images.

    Because the backbone is called twice per forward pass the hooks collect
    two entries each:
      activations[0] = pre  (first forward call)
      activations[1] = post (second forward call)
      gradients[0]   = post (backward processes last-first)
      gradients[1]   = pre
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def generate(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_climate: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        target_image: str = 'post',
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute a Grad-CAM heatmap for one satellite image.

        Parameters
        ----------
        target_image : 'pre' or 'post'

        Returns
        -------
        cam : np.ndarray  (H_feat, W_feat), values ∈ [0, 1]
        att_weights : np.ndarray  (T,)  — empty array if model has no attention
        target_class : int
        """
        self.activations.clear()
        self.gradients.clear()
        self.model.eval()

        model_out = self.model(x_pre, x_post, x_climate, event_labels)
        if isinstance(model_out, tuple):
            logits, att_weights_t = model_out
            att_weights = att_weights_t.detach().cpu().numpy()[0]
        else:
            logits = model_out
            att_weights = np.array([])

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        act_idx = 1 if target_image == 'post' else 0
        grad_idx = 0 if target_image == 'post' else 1

        acts = self.activations[act_idx][0]   # (C, H, W)
        grads = self.gradients[grad_idx][0]   # (C, H, W)

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=0))

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy(), att_weights, target_class


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _to_rgb(
    tensor: torch.Tensor,
    patch_mean: Optional[np.ndarray],
    patch_std: Optional[np.ndarray],
) -> np.ndarray:
    """Convert a (1, 3, H, W) tensor to an (H, W, 3) float32 array in [0, 1]."""
    img = tensor[0].detach().permute(1, 2, 0).cpu().numpy()
    if patch_mean is not None and patch_std is not None:
        img = (img * 3.0 * patch_std.reshape(1, 1, -1) + patch_mean.reshape(1, 1, -1))
        img = img.clip(0, 255).astype(np.uint8) / 255.0
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


def visualize_grad_cam(
    model: torch.nn.Module,
    x_pre: torch.Tensor,
    x_post: torch.Tensor,
    x_climate: torch.Tensor,
    event_labels: Optional[torch.Tensor],
    save_path: str,
    patch_mean: Optional[np.ndarray] = None,
    patch_std: Optional[np.ndarray] = None,
    damage_classes: Optional[dict] = None,
    gt_label: Optional[int] = None,
    sample_name: str = '',
) -> int:
    """
    Run Grad-CAM on pre and post images and write a figure to *save_path*.

    Returns the predicted class index.
    """
    target_layer = get_target_layer(model)
    grad_cam = MultimodalGradCAM(model, target_layer)

    cam_post, att_weights, pred_class = grad_cam.generate(
        x_pre, x_post, x_climate, event_labels, target_image='post'
    )
    cam_pre, _, _ = grad_cam.generate(
        x_pre, x_post, x_climate, event_labels,
        target_class=pred_class, target_image='pre',
    )
    grad_cam.remove_hooks()

    img_pre = _to_rgb(x_pre, patch_mean, patch_std)
    img_post = _to_rgb(x_post, patch_mean, patch_std)

    cam_pre_r = cv2.resize(cam_pre, (img_pre.shape[1], img_pre.shape[0]))
    cam_post_r = cv2.resize(cam_post, (img_post.shape[1], img_post.shape[0]))

    has_att = len(att_weights) > 0
    ncols = 3 if has_att else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(img_pre)
    axes[0].imshow(cam_pre_r, cmap='jet', alpha=0.45)
    axes[0].set_title('Pre-Disaster Grad-CAM')
    axes[0].axis('off')

    axes[1].imshow(img_post)
    axes[1].imshow(cam_post_r, cmap='jet', alpha=0.45)
    axes[1].set_title('Post-Disaster Grad-CAM')
    axes[1].axis('off')

    if has_att:
        axes[2].bar(np.arange(len(att_weights)), att_weights, color='teal')
        axes[2].set_title('Climate Temporal Attention')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Attention Weight')

    class_names = list(damage_classes.keys()) if damage_classes else None
    pred_name = class_names[pred_class] if class_names else str(pred_class)
    if gt_label is not None:
        gt_name = class_names[gt_label] if class_names else str(gt_label)
        title = f'{sample_name} — GT: {gt_name}  Pred: {pred_name}'
    else:
        title = f'{sample_name} — Pred: {pred_name}'

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return pred_class
