(* Require Import CoqHammer.Hammer. *)

(* Demo of proofs needing only simpl, reflexivity, auto, trivial, firstorder *)

Theorem double_neg : forall P : Prop, P -> ~~P.
Proof.
    firstorder. (*auto. will also work, but trivial won't work*)
Qed.

Theorem trival_implication : forall P : Prop, P -> P.
Proof.
    firstorder. (*auto. or trivial. will also work*)
Qed.

Theorem modus_ponens : forall P Q : Prop, P -> (P -> Q) -> Q.
Proof.
    firstorder. (*auto. will also work*)
    (*one can optionally do an intro and then first order*)
Qed.

Theorem modus_tollens : forall P Q : Prop, ~Q -> (P -> Q) -> ~P.
Proof.
    firstorder. (*auto. will also work*)
Qed.

Theorem disjunctive_syllogism : forall P Q : Prop, P \/ Q -> ~P -> Q.
Proof.
    firstorder. (*auto. will also work*)
Qed.

Theorem contrapositive : forall P Q : Prop, (P -> Q) -> ~Q -> ~P.
Proof.
    firstorder. (*auto. will also work*)
Qed.

Theorem nat_zero_add : forall n : nat, 0 + n = n.
Proof.
    (* simpl. reflexivity. *)
    (* auto. *)
    firstorder. (*auto. will also work*)
Qed.

Theorem nat_add_zero : forall n : nat, n + 0 = n.
Proof.
    (* auto. *)
    firstorder. (*auto. will also work*)
Qed.

Theorem nat_add_succ : forall n m : nat, n + S m = S (n + m).
Proof.
    (* auto. *)
    firstorder. (*auto. will also work*)
Qed.

Theorem nat_succ_add : forall n m : nat, S n + m = S (n + m).
Proof.
    (* trivial. *)
    (* intro. simpl. reflexivity. *)
    (* auto. *)
    firstorder. (*auto. will also work*)
Qed.

Theorem nat_add_comm : forall n : nat, n + 1 = 1 + n.
Proof.
    induction n.
    - simpl. reflexivity.
    - simpl. rewrite IHn. reflexivity.
Qed.