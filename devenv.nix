{ pkgs, lib, config, inputs, ... }:

{
  languages.python = {
    enable = true;
    version = "3.11";
    uv.enable = true;
    uv.sync.enable = true;
    venv.enable = true;
  };

  packages = with pkgs; [
    cowsay
    python311Packages.tkinter
    zlib
    arduino-cli
  ];
}