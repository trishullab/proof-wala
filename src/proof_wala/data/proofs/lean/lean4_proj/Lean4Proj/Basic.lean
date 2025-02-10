namespace simple

theorem n_plus_zero (n : Nat) : n + 0 = n := by
  sorry

theorem n_plus_succ (n m : Nat) : n + Nat.succ m = Nat.succ (n + m) := by
  sorry

theorem n_zero_plus (n : Nat) : 0 + n = n := by
  sorry

theorem n_succ_plus (n m : Nat) : Nat.succ n + m = Nat.succ (n + m) := by
  sorry

theorem n_mod_2_implies_square_mod (n : Nat) :
  n % 2 = 0 → n * n % 2 = 0 := by
  sorry

end simple
