(* Require Import CoqHammer.Hammer. *)
Require Import Coq.Arith.Arith.
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

Theorem finite_unary_functions : forall f : bool -> bool, 
(forall x : bool, (f x) = true) \/ 
(forall x : bool, (f x) = false) \/ 
(forall x : bool, (f x) = x) \/ 
(forall x : bool, (f x) = (negb x)).
Proof.
    intros f.
    destruct (f true) eqn:H_true; destruct (f false) eqn:H_false.
    left; intros; destruct x; try rewrite H_true; try rewrite H_false; reflexivity.
    right; right; left; intros; destruct x; try rewrite H_true; try rewrite H_false; reflexivity.
    right; right; right; intros; destruct x; try rewrite H_true; try rewrite H_false; reflexivity.
    right; left; intros; destruct x; try rewrite H_true; try rewrite H_false; reflexivity.
Qed.

Theorem mod_4_arith: forall n : nat, n mod 4 = 2 -> n * n mod 4 = 0.
Proof.
    intros.
    rewrite Nat.mul_mod; auto.
    rewrite H. auto.
Qed.

(* Proof by GPT-4 *)
(*
Proof.
  intros f.
  destruct (f true) eqn:Heq1, (f false) eqn:Heq2.
  - left. intros x. destruct x; assumption.
  - right; right; right. intros x. destruct x.
    + assumption.
    + simpl in Heq2. assumption.
  - right; left. intros x. destruct x.
    + assumption.
    + simpl in Heq2. assumption.
  - right; right; left. intros x. destruct x.
    + simpl. assumption.
    + assumption.
Qed. *)
