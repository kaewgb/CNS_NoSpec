CC			= g++
NVCC		= nvcc

GPU			= gpu
OBJ_FILES	= advance.o advance_cpu.o advance_hybrid.o ctoprim.o diffterm.o diffterm_cpu.o hypterm.o hypterm_cpu.o hypterm_merged.o helper_functions.o
HEADERS		= header.h helper_functions.h
THRUST		= $(HOME)/software/

CPU			= cpu


BENCHMARK	= advance
B_OBJ_FILES	= benchmark.o advance_cpu.o ctoprim.o diffterm.o diffterm_cpu.o hypterm.o hypterm_cpu.o hypterm_merged.o helper_functions.o

NVCCFLAGS	= -I$(THRUST) -arch=sm_20 --fmad=false --ptxas-options=-v --disable-warnings

$(GPU):			main.o $(OBJ_FILES)
				$(NVCC) -o $@ $^
$(CPU):			cpu.o $(OBJ_FILES)
				$(NVCC) -o $@ $^
$(BENCHMARK):	$(B_OBJ_FILES)
				$(NVCC) -o $@ $^

ddiff:		ddiff.cpp
			$(CC) $(CFLAGS) -o $@ $<
%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
%.o:		%.cu $(HEADERS)
			$(NVCC) -c $< -o $@ $(NVCCFLAGS)
clean:
			rm -f $(OBJ_FILES) $(TARGET)
