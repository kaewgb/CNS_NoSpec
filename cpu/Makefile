CC			= g++
CFLAGS		= -O2
TARGET		= cpu
OBJ_FILES	= main.o ctoprim.o hypterm.o diffterm.o helper_functions.o advance.o
HEADERS		= header.h helper_functions.h

$(TARGET):	$(OBJ_FILES)
			$(CC) -o $(TARGET) $^

ddiff:		ddiff.o
			$(CC) -o $@ $^

flops:		flops.o helper_functions.o
			$(CC) -o $@ $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
clean:
			rm -f $(OBJ_FILES) $(TARGET)
