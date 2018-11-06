all:
	gcc src/lodepng.c src/convolve.c -O0 -o convolve -lopenblas -I/home/edyounis/Workspace/libpfc/include -lpfc
	gcc src/lodepng.c src/im2col.c -O0 -o im2col -lopenblas -I/home/edyounis/Workspace/libpfc/include -lpfc
	gcc src/lodepng.c src/cacnn.c -O0 -o cacnn -lopenblas -I/home/edyounis/Workspace/libpfc/include -lpfc

