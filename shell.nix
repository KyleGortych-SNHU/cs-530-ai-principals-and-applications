/*
  shell.nix
  Development environment for the Module 5 Pygame project.
  Date: 10/25/2026
  Author: Kyle Gortych

  Contract:
    Preconditions: Nix and a nixpkgs channel are available.
    Postconditions: The shell provides Python 3.13, original Pygame, and pytest.
    Invariants: Python and Pygame versions match the project requirements.
    Effects: Exposes development dependencies without modifying project files.
*/
{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313.withPackages (pythonPackages: [
    pythonPackages.pygame
    pythonPackages.pytest
  ]);
in
pkgs.mkShell {
  packages = [
    python
  ];
}
