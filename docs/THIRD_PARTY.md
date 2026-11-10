# Third-party inputs

The ten PNGs in `assets/occluder/` are the fixed texture bank from the experiment: one clean LIBERO-Spatial frame per task. The corruption function blurs, darkens, and composites these inputs with a seeded silhouette. Replacing them changes the experiment.

LIBERO source: https://github.com/Lifelong-Robot-Learning/LIBERO

LIBERO's license notice is reproduced in `LIBERO_LICENSE.txt`. LIBERO incorporates simulator and visual assets from other projects; their rights remain with their respective owners. These frames do not grant a license to redistribute model weights or the original asset collections.

Isaac-GR00T is an external dependency under its upstream Apache-2.0 license. The distribution contains no modified NVIDIA sources or patches. Public measurement servers use hooks on the official source revision specified in `config/reproduction/gr00t_source.json`; checkpoint licenses remain separate.
