{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "qwen-image-env";
  targetPkgs = pkgs: (with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    git
    git-lfs
    which
    procps
    stdenv.cc.cc.lib
    zlib
    zstd
    numactl
    libGL
    libGLU
    glib
    libdrm
    vulkan-loader
    vulkan-tools
  ]);
  profile = ''
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export ROCM_PATH=/run/current-system/sw
    export HIP_VISIBLE_DEVICES=0
    export ROCR_VISIBLE_DEVICES=0
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
    if [ -d ".venv" ]; then
      source .venv/bin/activate
    fi
  '';
  runScript = "bash";
})
