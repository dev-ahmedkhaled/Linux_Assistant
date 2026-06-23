{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  },
}:

let
  unstable = import <unstable> { config.allowUnfree = true; };
  llama-cpp-cuda = unstable.llama-cpp.override { cudaSupport = true; };

in

pkgs.mkShell {
  name = "LinuxAssistant";

  buildInputs = with pkgs; [
    uv
    python312
    python312Packages.pip
    python312Packages.ipykernel
    python312Packages.ipywidgets

    # vllm
    llama-cpp-cuda

    cudaPackages.cudatoolkit

    gcc
    ninja

    zlib
    glibc.bin
    libGL
    stdenv.cc.cc.lib
  ];

  shellHook = ''
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  uv        : $(uv --version)"
    echo "  Python    : $(python --version)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


    # Nix
    export NIX_LD_LIBRARY_PATH="/run/opengl-driver/lib''${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}"
    export NIX_LD_LIBRARY_PATH="$CUDA_HOME/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$NIX_LD_LIBRARY_PATH"
    export NIX_SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib/

    #LD
    export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH"

    # CUDA
    export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
    export PATH="$CUDA_HOME/bin:$PATH"

    # Optional: remove these if you want pure Unsloth kernels instead of llama.cpp (didn't work)
    export UNSLOTH_USE_LLAMA_CPP_PYTHON=1
    export UNSLOTH_SKIP_LLAMA_CPP_INSTALL=1


    # Create / activate venv
    if [ ! -d ".venv" ]; then
      echo "→ Creating venv with uv..."
      uv venv .venv --python python3.12
    fi
    source .venv/bin/activate

    # Sanity check
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✅ CUDA is ready!')" 2>/dev/null || echo "⚠️  Install PyTorch: uv pip install torch --index-url https://download.pytorch.org/whl/cu128"

    # ── Fix 1: bitsandbytes / torch CUDA 13 runtime libs from pip wheels ──
    if python -c "import nvidia.nvjitlink" 2>/dev/null; then
      NVIDIA_BASE=$(python -c "import nvidia.nvjitlink; import pathlib; print(pathlib.Path(nvidia.nvjitlink.__file__).resolve().parent.parent)")
      for libdir in "$NVIDIA_BASE"/*/lib; do
        [ -d "$libdir" ] && export LD_LIBRARY_PATH="$libdir:$LD_LIBRARY_PATH"
      done
    fi

    # Sanity check
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✅ CUDA is ready!')" 2>/dev/null || echo "⚠️  Install PyTorch: uv pip install torch --index-url https://download.pytorch.org/whl/cu128"


  '';
}
