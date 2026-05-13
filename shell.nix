{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

pkgs.mkShell {
  name = "Optimization";

  buildInputs = with pkgs; [
    python312
    uv
    stdenv.cc.cc.lib
    zlib
    libGL
  ];

  shellHook = ''
    # ── Fix libstdc++.so.6 (and other native libs) not found at runtime ──
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libGL
    ]}:$LD_LIBRARY_PATH

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  uv   version: $(uv --version)"
    echo "  Python:       $(python --version)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -d ".venv" ]; then
      echo "→ Creating venv with uv..."
      uv venv .venv --python python3.12
    fi

    source .venv/bin/activate
    echo "→ Venv active: $VIRTUAL_ENV"
  '';
}