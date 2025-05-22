{ pkgs, lib, config, inputs, ... }:

{
  languages.python = {
    enable = true;
    version = "3.11";
    uv.enable = true;
    uv.sync.enable = true;
    venv.enable = true;
  };

  packages = [
    pkgs.cowsay
    pkgs.python311Packages.tkinter
    pkgs.zlib
  ];
}