import turtle
import math

edgeLength = 100
tortuga = turtle.Turtle()
tortuga.shape('turtle')
tortuga.right(90)
for i in range(4):
    tortuga.forward(edgeLength)
    tortuga.left(90)
    if i % 2 == 0:
        edgeLength = edgeLength*2
    else:
        edgeLength = edgeLength/2

tortuga.right(90)
tortuga.forward(30)
tortuga.right(135)
tortuga.forward(184)
tortuga.right(90)
tortuga.forward(184)
tortuga.right(135)
tortuga.forward(30)

turtle.done()