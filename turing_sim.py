import brain
import brain_util as bu
import numpy as np

def larger_k(n=10000,k=100,p=0.01,beta=0.05, bigger_factor=10):
	b = brain.Brain(p, save_winners=True)
	b.add_stimulus("stim", k)
	b.add_area("A",n,k,beta)
	b.add_area("B",n,bigger_factor*k,beta)
	b.update_plasticities(area_update_map={"A":[("B", 0.8), ("A", 0.0)],
											"B":[("A", 0.8), ("B", 0.8)]})
	b.project({"stim":["A"]},{})
	t=1
	while True:
		b.project({"stim":["A"]},{"A":["A"]})
		print "A total w is " + str(b.areas["A"].w) 
		if (b.areas["B"].num_first_winners <= 1) and (b.areas["A"].num_first_winners <= 1):
			print "proj(stim, A) stabilized after " + str(t) + " rounds"
			break
		t += 1
	A_after_proj = b.areas["A"].winners

	b.project({"stim":["A"]},{"A":["A","B"]})
	t=1
	while True:
		b.project({"stim":["A"]},{"A":["A","B"], "B":["B", "A"]})
		print("Num new winners in A " + str(b.areas["A"].num_first_winners))
		print("Num new winners in B " + str(b.areas["B"].num_first_winners))
		if (b.areas["B"].num_first_winners <= 1) and (b.areas["A"].num_first_winners <= 1):
			print "recip_project(A,B) stabilized after " + str(t) + " rounds"
			break
		t += 1
	print "Final statistics" 
	print "A.w = " + str(b.areas["A"].w)
	print "B.w = " + str(b.areas["B"].w)
	A_after_B = b.areas["A"].saved_winners[-1]
	o = bu.overlap(A_after_proj, A_after_B)
	print "Overlap is " + str(o)

def turing_erase(n=50000,k=100,p=0.01,beta=0.05, r=1.0, bigger_factor=20):
	b = brain.Brain(p, save_winners=True)
	# Much smaller stimulus, similar to lower p from stimulus into A
	smaller_k = int(r*k)
	b.add_stimulus("stim", smaller_k)
	b.add_area("A",n,smaller_k,beta)
	b.add_area("B",n,bigger_factor * k,beta)
	b.add_area("C",n,bigger_factor * k,beta)
	b.update_plasticities(area_update_map={"A":[("B", 0.8),("C", 0.8), ("A", 0.0)],
											"B":[("A", 0.8), ("B", 0.8)],
											"C":[("A", 0.8), ("C", 0.8)]},
						 stim_update_map={"A":[("stim", 0.05)]})
	b.project({"stim":["A"]},{})
	t=1
	while True:
		b.project({"stim":["A"]},{"A":["A"]})
		if (b.areas["B"].num_first_winners <= 1) and (b.areas["A"].num_first_winners <= 1):
			print "proj(stim, A) stabilized after " + str(t) + " rounds"
			break
		t += 1

	b.project({"stim":["A"]},{"A":["A","B"]})
	t=1
	while True:
		b.project({"stim":["A"]},{"A":["A","B"], "B":["B", "A"]})
		print("Num new winners in A " + str(b.areas["A"].num_first_winners))
		if (b.areas["B"].num_first_winners <= 1) and (b.areas["A"].num_first_winners <= 1):
			print "recip_project(A,B) stabilized after " + str(t) + " rounds"
			break
		t += 1
	A_after_proj_B = b.areas["A"].winners 

	b.project({"stim":["A"]},{"A":["A","C"]})
	t=1
	while True:
		b.project({"stim":["A"]},{"A":["A","C"], "C":["C", "A"]})
		print("Num new winners in A " + str(b.areas["A"].num_first_winners))
		if (b.areas["C"].num_first_winners <= 1) and (b.areas["A"].num_first_winners <= 1):
			print "recip_project(A,C) stabilized after " + str(t) + " rounds"
			break
		t += 1
	A_after_proj_C = b.areas["A"].winners

	# Check final conditions
	b.project({},{"A":["B"]})
	B_after_erase = b.areas["B"].saved_winners[-1]
	B_before_erase =  b.areas["B"].saved_winners[-2]
	B_overlap = bu.overlap(B_after_erase, B_before_erase)
	print("Overlap of B after erase and with y is " + str(B_overlap) + "\n")
	A_overlap = bu.overlap(A_after_proj_B,A_after_proj_C)
	print("Overlap of A after proj(B) vs after proj(C) is " + str(A_overlap) + "\n")