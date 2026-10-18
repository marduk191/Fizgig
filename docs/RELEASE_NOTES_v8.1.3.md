# Fizgig v8.1.3: maintenance

- **Fine-tuning no longer shows Adaptive LR,** on any model. A fine-tune trains at its own learning rate, and an Adaptive LR tick left on from a LoRA setup no longer reaches the trainer.
- **Flux 3 Image Klein's Ultra Fast preset** now trains for 50 epochs.
