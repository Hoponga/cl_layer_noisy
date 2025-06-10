# Continual Anchored Manifold Embeddings for Learning Stability (CAMELS)

A latent-space geometry-inspired approach for rehearsal-based continual learning by enforcing per-task metric space isometries 

To run: 
```python train_ot_manifold_buffer.py```


For DDP sweep: 

torchrun --standalone --nproc_per_node=<NUM_GPUS> sweep_params.py
