Target		:= main

CC		:= g++
CXXFLAGS	:= -std=c++11 -Wall -g -O2

INC_DIR		:= -Iinclude -I../matrix/include 
LIBS		:= -L../matrix/lib -lMatrix 

SRCS		:= $(wildcard src/*.cpp)
OBJS		:= $(addprefix obj/, $(patsubst %.cpp,%.o,$(notdir ${SRCS})))
LIB_OBJ		:= lib/libcnn_core.so

${Target}: ${LIB_OBJ} obj/main.o
	${CC} ${CXXFLAGS} ${INC_DIR} ${LIBS} -Llib -lcnn_core -o $@ $^

${LIB_OBJ}: $(filter-out obj/main.o,${OBJS})
	ar cr $@ $^ 

obj/%.o:src/%.cpp
	${CC} ${CXXFLAGS} ${INC_DIR} ${LIB_DIR} -o $@ -c $^

clean:
	rm -rf ${Target} ${OBJS} ${LIB_OBJ}
