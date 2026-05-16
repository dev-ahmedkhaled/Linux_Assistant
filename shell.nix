{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

pkgs.mkShell {
<<<<<<< HEAD
  name = "Optimization";

  buildInputs = with pkgs; [
    python312
    uv
    stdenv.cc.cc.lib
=======
  name = "LinuxAssistant";

  buildInputs = with pkgs; [
    # Python + uv
    python312
    uv

    # Common native deps useful for ML/CUDA projects
    stdenv.cc.cc.lib   # libstdc++
>>>>>>> 4718d41 (added a shell)
    zlib
    libGL
  ];

  shellHook = ''
<<<<<<< HEAD
    # ── Fix libstdc++.so.6 (and other native libs) not found at runtime ──
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libGL
    ]}:$LD_LIBRARY_PATH

=======
>>>>>>> 4718d41 (added a shell)
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  uv   version: $(uv --version)"
    echo "  Python:       $(python --version)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

<<<<<<< HEAD
=======
    # Create venv with uv if it doesn't exist yet
>>>>>>> 4718d41 (added a shell)
    if [ ! -d ".venv" ]; then
      echo "→ Creating venv with uv..."
      uv venv .venv --python python3.12
    fi

<<<<<<< HEAD
    source .venv/bin/activate
    echo "→ Venv active: $VIRTUAL_ENV"
  '';
}
=======
    # Activate it
    source .venv/bin/activate
  '';
}
>>>>>>> 4718d41 (added a shell)
