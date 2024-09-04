opam switch create MathComp 4.14.1
eval $(opam env --switch=MathComp --set-switch)
opam repo add coq-released https://coq.inria.fr/opam/released
opam pin add -y coq 8.18.0