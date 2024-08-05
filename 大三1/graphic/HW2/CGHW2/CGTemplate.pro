QT += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += console qt c++11

DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += "./glm"

INCLUDEPATH += "E:\Qt\glew-2.2.0\include"



LIBS += \
	Glu32.lib \
	OpenGL32.lib
LIBS += glew32.lib

LIBPATH += "E:\Qt\glew-2.2.0\lib\Release\x64"

SOURCES += \
    main.cpp \
    myglwidget.cpp \
    utils.cpp

HEADERS += \
    myglwidget.h  \
    utils.h
