FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<

pNbody: nbody.cu compute.cu
	nvcc $(FLAGS) nbody.cu compute.cu -o pNbody $(LIBS)


clean:
	rm -f *.o nbody pNbody *.err *.out
