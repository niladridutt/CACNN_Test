all:
	mkdir -p build
	mkdir -p lib
	gcc -fopenmp src/cacnn.c -O3 -Iinclude -c -o build/cacnn.o
	export OMP_NUM_THREADS=2
	gcc src/convolve.c -O3 -Iinclude -c -o build/convolve.o
	gcc src/carma.c -O3 -Iinclude -c -o build/carma.o -lcblas
	ar rcs lib/libcacnn.a build/cacnn.o
	ar rcs lib/libconvolve.a build/convolve.o
	ar rcs lib/libcarma.a build/carma.o
	gcc -fopenmp src/main.c src/lodepng.c lib/* -Iinclude -I/home/edyounis/Workspace/applications/libpfc/include -lcblas -lm -O0 -o measure

