all:
	mkdir -p build
	mkdir -p lib
	gcc src/cacnn.c -O3 -Iinclude -c -o build/cacnn.o
	gcc src/convolve.c -O3 -Iinclude -c -o build/convolve.o
	ar rcs lib/libcacnn.a build/cacnn.o
	ar rcs lib/libconvolve.a build/convolve.o
	gcc src/main.c src/lodepng.c lib/* -Iinclude -I/home/edyounis/Workspace/applications/libpfc/include -lcblas -lm -lpfc -O0 -o measure

