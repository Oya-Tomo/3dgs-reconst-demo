{
  description = "Nerfstudio 3DGS reconstruction development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;
      runtimeLibraries = with pkgs; [
        glib
        libglvnd
        stdenv.cc.cc.lib
        zlib
      ];
      runtimeLibraryPath = pkgs.lib.makeLibraryPath runtimeLibraries;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cmake
          ffmpeg
          gcc
          git
          gnumake
          ninja
          pkg-config
          python
          uv
        ];

        UV_PYTHON = "${python}/bin/python3.12";
        UV_PYTHON_DOWNLOADS = "never";

        # Prebuilt Spectacular AI and OpenCV wheels rely on these dynamic
        # libraries without carrying Nix-aware runtime search paths. Nix's
        # loader also does not search the host's FHS NVIDIA driver paths. Link
        # only libcuda into an isolated directory: adding the complete host
        # library directory would mix the host and Nix glibc implementations.
        shellHook = ''
          projectDriverLinkDir="$PWD/.direnv/host-driver-libs"
          mkdir -p "$projectDriverLinkDir"

          for projectDriverCandidate in \
            /run/opengl-driver/lib/libcuda.so.1 \
            /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
            /usr/lib64/libcuda.so.1 \
            /usr/lib/wsl/lib/libcuda.so.1
          do
            if [ -e "$projectDriverCandidate" ]; then
              projectDriverLibrary="$(readlink -f "$projectDriverCandidate")"
              ln -sfn "$projectDriverLibrary" "$projectDriverLinkDir/libcuda.so.1"
              ln -sfn "$projectDriverLibrary" "$projectDriverLinkDir/libcuda.so"
              break
            fi
          done

          export LD_LIBRARY_PATH="${runtimeLibraryPath}:$projectDriverLinkDir''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
