CC			= g++
CFLAGS		=
TARGET		= advance
OBJ_FILES	= main.o ctoprim.o hypterm.o diffterm.o helper_functions.o
HEADERS		= header.h helper_functions.h

$(TARGET):	$(OBJ_FILES)
			$(CC) -o $(TARGET) $^

%.o:		%.cpp $(HEADERS)
			$(CC) $(CFLAGS) -c $< -o $@
clean:
			rm -f $(OBJ_FILES) $(TARGET)
