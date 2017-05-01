ifndef NVCXX
NVCXX := nvcc
endif

ifndef PNGPP
PNGPP := /usr/local/Cellar/png++/0.2.5_1
endif

ifndef PREFIX
PREFIX := /usr/local
endif

ifndef LIBPNG
LIBPNG := /usr/local/Cellar/libpng/1.6.29
endif

ifndef LIBPNG_CONFIG
LIBPNG_CONFIG := libpng-config
endif

all:
	$(NVCXX) -o render render.cu -Xlinker -framework,OpenGL,-framework,GLUT  -I$(PREFIX)/include -I$(LIBPNG)/include/libpng16 -I$(PNGPP)/include/png++ -L$(PREFIX)/lib -L$(LIBPNG)/lib -lpng16

clean:
	rm render