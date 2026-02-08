"""
Landscape probe implementations for fine-tuning analysis.
Measures: gradient statistics, SAM sharpness, Hutchinson trace, top eigenvalue.
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Dict, Tuple
from contextlib import contextmanager
import numpy as np


@contextmanager
def disable_sdpa():
    """
    Context manager to disable scaled dot-product attention.
    SDPA doesn't support second-order gradients needed for Hessian computation.
    """
    # Save original states
    flash_enabled = torch.backends.cuda.flash_sdp_enabled()
    mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_enabled = torch.backends.cuda.math_sdp_enabled()

    try:
        # Disable efficient implementations, keep only math (supports double backward)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        # Restore original states
        torch.backends.cuda.enable_flash_sdp(flash_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient_enabled)
        torch.backends.cuda.enable_math_sdp(math_enabled)


class LandscapeProbes:
    """Compute cheap landscape probes for a model at a given point."""

    def __init__(self, model: nn.Module, criterion: nn.Module, device: str = 'cuda'):
        self.model = model
        self.criterion = criterion
        self.device = device

    def compute_all_probes(
        self,
        dataloader,
        n_batches: int = 5,
        n_hutchinson_samples: int = 10,
        sam_rho: float = 0.05
    ) -> Dict[str, float]:
        """
        Compute all probes using the first n_batches of data.

        Returns dict with:
            - gradient_norm: mean gradient L2 norm
            - gradient_variance: variance of gradient norms across batches
            - sam_sharpness: L(w + epsilon) - L(w) where epsilon is SAM perturbation
            - hutchinson_trace: estimated Hessian trace
            - top_eigenvalue: largest Hessian eigenvalue (via power iteration)
        """
        probes = {}

        # Collect batches
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("Dataloader is empty, cannot compute probes")

        # 1. Gradient statistics (cheap)
        grad_norms = []
        losses = []

        for batch in batches:
            self.model.zero_grad()
            inputs, labels = self._unpack_batch(batch)
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()

            # Collect gradient norm
            grad_vec = self._get_grad_vector()
            grad_norms.append(grad_vec.norm().item())
            losses.append(loss.item())

        probes['gradient_norm'] = np.mean(grad_norms)
        probes['gradient_variance'] = np.var(grad_norms)
        probes['loss_at_init'] = np.mean(losses)

        # 2. SAM sharpness (1 extra forward pass)
        probes['sam_sharpness'] = self._compute_sam_sharpness(batches[0], sam_rho)

        # 3. Hutchinson trace (n_hutchinson_samples Hessian-vector products)
        probes['hutchinson_trace'] = self._compute_hutchinson_trace(
            batches[0], n_samples=n_hutchinson_samples
        )

        # 4. Top eigenvalue (power iteration)
        probes['top_eigenvalue'] = self._compute_top_eigenvalue(
            batches[0], n_iterations=20
        )

        return probes

    def _unpack_batch(self, batch) -> Tuple[Dict, torch.Tensor]:
        """Unpack batch for HuggingFace models."""
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        return inputs, labels

    def _get_grad_vector(self) -> torch.Tensor:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def _get_param_vector(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        params = []
        for p in self.model.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)

    def _set_param_vector(self, vec: torch.Tensor):
        """Set parameters from a flattened vector."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = vec[offset:offset + numel].view(p.shape)
            offset += numel

    def _compute_sam_sharpness(self, batch, rho: float = 0.05) -> float:
        """
        Compute SAM-style sharpness: max_{||e||<=rho} L(w+e) - L(w)
        Approximated by: L(w + rho * grad/||grad||) - L(w)
        """
        self.model.zero_grad()
        inputs, labels = self._unpack_batch(batch)

        # Compute base loss and gradient
        outputs = self.model(**inputs)
        base_loss = self.criterion(outputs.logits, labels)
        base_loss.backward()

        grad_vec = self._get_grad_vector()
        grad_norm = grad_vec.norm()

        if grad_norm < 1e-12:
            return 0.0

        # Save original parameters
        orig_params = self._get_param_vector().clone()

        # Perturb in gradient direction
        epsilon = rho * grad_vec / grad_norm
        self._set_param_vector(orig_params + epsilon)

        # Compute perturbed loss
        with torch.no_grad():
            outputs = self.model(**inputs)
            perturbed_loss = self.criterion(outputs.logits, labels)

        # Restore parameters
        self._set_param_vector(orig_params)

        return (perturbed_loss - base_loss).item()

    def _hessian_vector_product(self, batch, vec: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product Hv using autodiff.
        Uses the identity: Hv = d/dt [grad L(w + t*v)] at t=0
        """
        # Disable SDPA as it doesn't support second-order gradients
        with disable_sdpa():
            self.model.zero_grad()
            inputs, labels = self._unpack_batch(batch)

            # Forward pass
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, labels)

            # First gradient
            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = grad(loss, params, create_graph=True)

            # Flatten gradients
            grad_vec = torch.cat([g.view(-1) for g in grads])

            # Compute Hv = d(grad . v)/d(params)
            grad_dot_v = (grad_vec * vec).sum()
            hvp = grad(grad_dot_v, params, retain_graph=False)

            return torch.cat([h.view(-1) for h in hvp])

    def _compute_hutchinson_trace(self, batch, n_samples: int = 10) -> float:
        """
        Estimate tr(H) using Hutchinson's estimator:
        tr(H) ≈ (1/k) sum_i v_i^T H v_i
        where v_i are Rademacher random vectors.
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        trace_estimates = []

        for _ in range(n_samples):
            # Rademacher random vector (±1 with equal probability)
            v = torch.randint(0, 2, (n_params,), device=self.device).float() * 2 - 1

            # Compute Hv
            Hv = self._hessian_vector_product(batch, v)

            # v^T H v
            trace_estimates.append((v * Hv).sum().item())

        return np.mean(trace_estimates)

    def _compute_top_eigenvalue(self, batch, n_iterations: int = 20) -> float:
        """
        Estimate top Hessian eigenvalue using power iteration.
        """
        n_params = sum(p.numel() for p in self.model.parameters())

        # Random initial vector
        v = torch.randn(n_params, device=self.device)
        v = v / v.norm()

        eigenvalue = 0.0
        for _ in range(n_iterations):
            # Hv
            Hv = self._hessian_vector_product(batch, v)

            # Rayleigh quotient estimate
            eigenvalue = (v * Hv).sum().item()

            # Normalize
            v = Hv / (Hv.norm() + 1e-12)

        return eigenvalue


def compute_probes(model, criterion, dataloader, device='cuda') -> Dict[str, float]:
    """One-liner to compute all probes."""
    prober = LandscapeProbes(model, criterion, device)
    return prober.compute_all_probes(dataloader)
