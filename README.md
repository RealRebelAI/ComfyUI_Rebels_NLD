# ComfyUI_Rebels_NLD -- NODES ARE OPERATIONAL BUT SLOW. Currently working on patches for speed ups. they will run in their current state but i recommend git pulling frequently. FLASH ATTENTION HELPS DRAMATICALLY.

GGUF loader + text-to-image nodes for NVIDIA **NL-Diffusion-Image** (masked discrete diffusion
LM + IBQ VQ decoder) on consumer hardware. By RealRebelAI.


## Install

1. Clone into `ComfyUI/custom_nodes/`.
2. Requires the **city96 ComfyUI-GGUF fork** in the same `custom_nodes/` folder (used for dequant).
3. Put the model files (dropdown-selected, no paths):
   - **dLM GGUF** → `ComfyUI/models/unet/`
   - **vqvae** (bf16 `.safetensors`) → `ComfyUI/models/vae/`
4. The config/tokenizer/modeling code ships in `model_assets/`

## IMPORTANT
5. model.safetensors file MUST go in "custom_nodes\ComfyUI_Rebels_NLD\model_assets\emu3_vqvae"

https://huggingface.co/nvidia/NL-Diffusion-Image/blob/main/emu3_vqvae/model.safetensors

## Flash Attention Wheels for Windows Users (speed up gen time)
youll have to do the work to install based on your environment. - https://huggingface.co/Wildminder/AI-windows-whl/tree/main

## Nodes

- **NL-Diffusion dLM Loader (GGUF)** — pick `gguf_name` and `vqvae_name` from dropdowns, choose device.
- **NL-Diffusion Text to Image** — prompt, size, steps, guidance, temperature, seed → IMAGE.

## Notes

- The dLM generates discrete token indices; the vqvae decoder turns them into pixels. It is **not**
  a latent VAE — it loads through this pack, not ComfyUI's VAELoader.
- Vocab embeddings use row-gather dequant, so the 131k-row tensors never fully materialize.
- Vision-tower (image-understanding / edit) weights are left on meta and not needed for t2i.

## License

Model is under the **NVIDIA One-Way Noncommercial License** (research/development only). Quants
inherit those terms — publish as `license: other` with the upstream terms linked.
