simple_code_gprof: simple_code.c
	gcc $< -march=native -fno-inline -O3 -pg -o $@

N?=100000
.PHONY: gmon.out
gmon.out: simple_code_gprof
	./$< $N

gprof.out: simple_code_gprof gmon.out
	gprof $^ > $@

simple_code_likwid: simple_code_likwid.c
	gcc -std=c99 -mfma -O3 -DLIKWID_PERFMON -fno-inline -march=native \
		-o $@ $^ -llikwid

GROUP?="DATA"
run_likwid: simple_code_likwid
	likwid-perfctr -m -g $(GROUP) -C 0 ./simple_code_likwid $N

clean:
	rm -rf gmon.out ./simple_code ./simple_code_gprof ./simple_code_likwid
