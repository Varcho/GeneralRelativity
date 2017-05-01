all:
	nvcc -o render render.cu -Xlinker -framework,OpenGL,-framework,GLUT

clean:
	rm render
