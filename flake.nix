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
        # libraries without carrying Nix-aware runtime search paths.
        shellHook = ''
          export LD_LIBRARY_PATH="${runtimeLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
