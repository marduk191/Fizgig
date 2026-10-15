# Fizgig v7.2.3: maintenance

Qwen Image 2.1's Profiler and Repair Studio load the current Turbo int8 file again.

## Fixes

- **Qwen Image 2.1 Turbo int8 file (#184).** Comfy-Org re-uploaded `qwen_image_2.1_turbo_int8_convrot.safetensors` on 9 Oct with new per-block attention settings, and the Profiler, Repair Studio and other workbench tabs stopped with a "DiT keys mismatch" error when loading it. Both the new file and earlier downloads now load. Thanks to @Emerganz for the report.

To update, run `update_fizgig.bat` (`update_fizgig_rocm.bat` on AMD). Pods pick up the update on their next start.
