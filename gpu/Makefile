CC			= g++
NVCC		= nvcc


CPU			= cpu
CPU_OBJS	= advance_cpu.o diffterm_cpu.o hypterm_cpu.o util_cpu.o ctoprim.o util.o

GPU			= gpu
GPU_OBJS	= advance.o ctoprim.o diffterm2.o hypterm2.o util.o $(CPU_OBJS)
HEADERS		= header.h util.h util.cuh
THRUST		= $(HOME)/software/

HYBRID		= hybrid


CFLAGS		=
NVCCFLAGS	= -I$(THRUST) -arch=sm_20 --fmad=false --ptxas-options=-v --disable-warnings

$(GPU):			main.o $(GPU_OBJS)
				$(NVCC) -o $@ $^
$(CPU):			cpu.o $(CPU_OBJS)
				$(NVCC) -o $@ $^
$(HYBRID):		hybrid.o $(GPU_OBJS) advance_hybrid.o
				$(NVCC) -o $@ $^
test:		test.o util.o util_cpu.o
			$(NVCC) -o $@ $^

ddiff:		ddiff.cpp
			$(CC) $(CFLAGS) -o $@ $<
extract:	extract.o util_cpu.o
			$(CC) $(CFLAGS) -o $@ $^
convert:	convert.o util_cpu.o
			$(CC) $(CFLAGS) -o $@ $^
3d:			3d.cu
			$(NVCC) -o $@ $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
%.o:		%.cu $(HEADERS)
			$(NVCC) -c $< -o $@ $(NVCCFLAGS)
clean:
			rm -f $(GPU_OBJS) $(CPU_OBJS) gpu cpu
